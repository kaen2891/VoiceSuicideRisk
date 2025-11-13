from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from util.dataset import PsychiatryDataset
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from transformers import AutoModel, AutoProcessor
from transformers import Wav2Vec2Model, HubertModel, AutoFeatureExtractor

import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save') 
    parser.add_argument('--save_name', type=str, default='./save') 
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')
    # dataset
    parser.add_argument('--dataset', type=str, default='psychiatry')
    parser.add_argument('--data_folder', type=str, default='./dataset/')
    parser.add_argument('--annotation', type=str, default='/home/jovyan/speech/dataset/fold_split_full/fold_1.csv')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='psychiatry',
                        help='psychiatry: (before, after)')
    parser.add_argument('--n_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--m_cls', type=int, default=0,
                        help='set k-way classification problem for domain (meta)')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--desired_length', type=int,  default=5, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--divide_length', type=int,  default=5, 
                        help='fixed length size of individual cycle')

    # model
    parser.add_argument('--model', type=str,
                        default='facebook/wav2vec2-base',
                        choices=['facebook/wav2vec2-base', 'facebook/hubert-base-ls960', 'microsoft/wavlm-base-plus'])
    parser.add_argument('--task_group', type=str, default='all',
                    choices=['all', 'Incongruent', 'Color', 'Word'],
                    help='subset task to use (default: all)')
    
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    parser.add_argument('--method', type=str, default='ce')
    
    
    parser.add_argument('--domain_adaptation', action='store_true')
    parser.add_argument('--domain_adaptation2', action='store_true')
    parser.add_argument('--cut_test', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.06)
    parser.add_argument('--target_type', type=str, default='project1_project2block', help='how to make target representation',
                        choices=['project_flow_all', 'representation_all', 'z1block_project', 'z1_project2', 'project1block_project2', 'project1_r2block', 'project1_r2', 'project1_project2block', 'project_block_all'])
    # for RepAugment
    
                       
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)
    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    if args.dataset == 'psychiatry':
        if args.class_split == 'psychiatry':  
            if args.n_cls == 2:
                args.cls_list = ['low', 'high']
            else:
                raise NotImplementedError

    return args

def seed_worker(worker_id):
    worker_seed = 1
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_loader(args):
    if args.dataset == 'psychiatry':
        train_dataset = PsychiatryDataset(train_flag=True, args=args, annotation=args.annotation, print_flag=True)
        val_dataset = PsychiatryDataset(train_flag=False, args=args, annotation=args.annotation, print_flag=True)
    else:
        raise NotImplemented
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker
    )
        
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)


    return train_loader, val_loader, args

from curses import meta
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaCL(nn.Module): 
    def __init__(self, temperature=0.06, weights=None): 
        super().__init__()
        self.temperature = temperature
        self.weights = weights

    def forward(self, projection1, projection2, meta_labels=None):

        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        features = torch.cat([projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1)
        batch_size = features.shape[0]

        if meta_labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        else:
            meta_labels = meta_labels.contiguous().view(-1, 1)
            mask = torch.eq(meta_labels, meta_labels.T).float().cuda()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        
        return loss

import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim=768, apply_bn=True):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU()
        if apply_bn:
            self.projector = nn.Sequential(self.linear1, self.bn, self.relu, self.linear2)
        else:
            self.projector = nn.Sequential(self.linear1, self.relu, self.linear2)

    def forward(self, x):
        return self.projector(x)

def set_model(args):    
    #model = PretrainedSpeechClassifier(args.model, args.n_cls)
    #model = AutoModel.from_pretrained(args.model)

    # 캐시 경로 지정
    CACHE_DIR = "/home/jovyan/.cache/huggingface/hub" # change!

    model = AutoModel.from_pretrained(
        args.model,
        cache_dir=CACHE_DIR,
        local_files_only=False
    )
    classifier = nn.Linear(768, args.n_cls)
    criterion = nn.CrossEntropyLoss()

    if args.domain_adaptation:
        criterion2 = nn.CrossEntropyLoss()
        criterion = [criterion.cuda(), criterion2.cuda()]
    elif args.domain_adaptation2:
        criterion2 = MetaCL(temperature=args.temperature)
        criterion = [criterion.cuda(), criterion2.cuda()]
    else:
        criterion = [criterion.cuda()]
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    if args.domain_adaptation:
        domain_classifier = nn.Linear(768, 2)
    elif args.domain_adaptation2:
        domain_classifier = Projector(768, 768)
    else:
        domain_classifier = nn.Identity()
    
    model.cuda()
    classifier.cuda()
    domain_classifier.cuda()
    
    optim_params = list(model.parameters()) + list(classifier.parameters()) + list(domain_classifier.parameters())
    optimizer = set_optimizer(args, optim_params)
    
    return model, classifier, domain_classifier.cuda(), criterion, optimizer

from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def feature_level_augment(features, time_mask_ratio=0.1, feature_mask_ratio=0.1):
    # SupCon only
    """
    features: [B, T, D]
    time_mask_ratio: fraction of time steps to mask
    feature_mask_ratio: fraction of feature dims to mask
    """
    B, T, D = features.shape
    masked = features.clone()

    # 🔹 Time masking (SpecAugment-style)
    time_mask_len = int(T * time_mask_ratio)
    if time_mask_len > 0:
        for b in range(B):
            t0 = np.random.randint(0, max(1, T - time_mask_len))
            masked[b, t0:t0 + time_mask_len, :] = 0

    # 🔹 Feature masking (channel-wise)
    feat_mask_len = int(D * feature_mask_ratio)
    if feat_mask_len > 0:
        for b in range(B):
            f0 = np.random.randint(0, max(1, D - feat_mask_len))
            masked[b, :, f0:f0 + feat_mask_len] = 0

    return masked  # shape [B, T, D]

def train(train_loader, model, classifier, domain_classifier, criterion, optimizer, epoch, args, scaler=None):
    model.train()
    classifier.train()
    if args.domain_adaptation or args.domain_adaptation2:
        domain_classifier.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels, genders) in enumerate(train_loader):
        # data load
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        if args.domain_adaptation or args.domain_adaptation2:
            if isinstance(genders, list):
                if isinstance(genders[0], str):
                    gender_map = {'M': 0, 'm': 0, 'F': 1, 'f': 1}
                    genders = [gender_map[g.lower()] for g in genders]
                genders = torch.tensor(genders, dtype=torch.long)
            genders = genders.cuda(non_blocking=True)
            domain_labels = genders
        bsz = labels.shape[0]
        
        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                if args.domain_adaptation or args.domain_adaptation2:
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(domain_classifier.state_dict())]
                    p = float(idx + epoch * len(train_loader)) / args.epochs + 1 / len(train_loader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                else:
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]
                    alpha = None

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            images = torch.squeeze(images, 1)
            outputs = model(images)
            hidden = outputs.last_hidden_state.mean(dim=1)
            logits = classifier(hidden)
            loss = criterion[0](logits, labels)

            if args.domain_adaptation:
                domain_output = ReverseLayerF.apply(hidden, alpha)
                domain_output = domain_classifier(domain_output)
                domain_loss = criterion[1](domain_output, domain_labels)
                loss += domain_loss
            
            if args.domain_adaptation2:
                outputs2 = model(images)
                # 🔹 feature-level augmentation (time + feature masking)
                features = feature_level_augment(outputs2.last_hidden_state, time_mask_ratio=0.1, feature_mask_ratio=0.1)
                hidden2 = features.mean(dim=1)
                domain_output = ReverseLayerF.apply(hidden2, alpha)
                
                if args.target_type == 'project1_project2block':
                    proj1 = domain_classifier(hidden)
                    proj2 = deepcopy(domain_classifier(domain_output).detach())

                domain_loss = criterion[1](proj1, proj2, domain_labels)
                loss += domain_loss

        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(logits[:bsz], labels, topk=(1,))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                # exponential moving average update
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
                if args.domain_adaptation or args.domain_adaptation2:
                    domain_classifier = update_moving_average(args.ma_beta, domain_classifier, ma_ckpt[2])
                classifier = update_moving_average(args.ma_beta, classifier, ma_ckpt[1])

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls

    all_preds = []
    all_labels = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, _) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            

            with torch.cuda.amp.autocast():
                images = torch.squeeze(images, 1)
                outputs = model(images)
                hidden = outputs.last_hidden_state.mean(dim=1)
                logits = classifier(hidden)
                loss = criterion[0](logits, labels)

            losses.update(loss.item(), bsz)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            
            [acc1], _ = accuracy(logits[:bsz], labels, topk=(1,))
            top1.update(acc1[0], bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
        
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy, Precision, Recall, F1-score
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary' if args.n_cls == 2 else 'macro')
    recall = recall_score(all_labels, all_preds, average='binary' if args.n_cls == 2 else 'macro')
    f1 = f1_score(all_labels, all_preds, average='binary' if args.n_cls == 2 else 'macro')

    print(f"\nEvaluation Results:")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall   : {recall * 100:.2f}%")
    print(f"  F1-Score : {f1 * 100:.2f}%\n")
    
    
    if f1 > best_acc[-1]:
        save_bool = True
        best_acc = [acc, precision, recall, f1]  # save all metrics
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]
        print(f"✅ Best model updated (F1 = {f1:.4f})")
    
    print("Best F1: {}".format(best_acc))

    return best_acc, best_model, save_bool

def validate_by_gender(val_loader, model, classifier, criterion, args):
    model.eval()
    classifier.eval()

    all_preds, all_labels, all_genders = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                images, labels, genders = batch
            else:
                images, labels = batch
                genders = ['unknown'] * len(labels)

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                images = torch.squeeze(images, 1)
                outputs = model(images)
                hidden = outputs.last_hidden_state.mean(dim=1)
                logits = classifier(hidden)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_genders.extend(genders)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_genders = np.array(all_genders)

    gender_groups = np.unique(all_genders)
    gender_results = {}

    for gender in gender_groups:
        idxs = np.where(all_genders == gender)[0]
        y_true, y_pred = all_labels[idxs], all_preds[idxs]

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary' if args.n_cls == 2 else 'macro')
        recall = recall_score(y_true, y_pred, average='binary' if args.n_cls == 2 else 'macro')
        f1 = f1_score(y_true, y_pred, average='binary' if args.n_cls == 2 else 'macro')

        gender_results[gender] = {
            "Accuracy": round(acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Samples": int(len(y_true))
        }

    print("\n====== Gender-wise Results ======")
    for g, m in gender_results.items():
        print(f"Gender: {g}")
        print(f"  #Samples : {m['Samples']}")
        print(f"  Accuracy : {m['Accuracy']*100:.2f}%")
        print(f"  Precision: {m['Precision']*100:.2f}%")
        print(f"  Recall   : {m['Recall']*100:.2f}%")
        print(f"  F1-Score : {m['F1']*100:.2f}%\n")
    print("=================================\n")

    return gender_results


def set_seed(seed): # for reproducibility
    import torch, numpy as np, random, os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    #torch.autograd.set_detect_anomaly(True)

    # fix seed
    set_seed(args.seed)
    
    best_model = None
    if args.dataset == 'psychiatry':
        best_acc = [0, 0, 0, 0]  # Acc, Precision, Recall, F1
    
    train_loader, val_loader, args = set_loader(args)
    model, classifier, domain_classifier, criterion, optimizer = set_model(args)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
    
    print('*' * 20)
     
    if not args.eval:
        print('Experiments {} start'.format(args.model_name))
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        
        for epoch in range(args.start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            
            loss, acc = train(train_loader, model, classifier, domain_classifier, criterion, optimizer, epoch, args, scaler)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2-time1, acc))
            
            # eval for one epoch
            #best_acc, best_model, save_bool = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
            best_acc, best_model, save_bool = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
            #acc, precision, recall, f1 = metrics
            # save a checkpoint of model and classifier when the best score is updated
            if save_bool:            
                save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                print('Best ckpt is modified with Acc = {} when Epoch = {}'.format(best_acc[0], epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier)

             


            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier)

        # save a checkpoint of classifier with the best accuracy or score
        save_file = os.path.join(args.save_folder, 'best.pth')
        model.load_state_dict(best_model[0])
        classifier.load_state_dict(best_model[1])
        save_model(model, optimizer, args, epoch, save_file, classifier)
    else:
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc, _, _  = validate(val_loader, model, classifier, criterion, args, best_acc)
    
    print("\n================ Final Best Results ================")
    print(f"Best Accuracy : {best_acc[0] * 100:.2f}%")
    print(f"Best Precision: {best_acc[1] * 100:.2f}%")
    print(f"Best Recall   : {best_acc[2] * 100:.2f}%")
    print(f"Best F1-Score : {best_acc[3] * 100:.2f}%")
    print("====================================================\n")

    print("\n========== Final Evaluation (Best Model) ==========")
    model.load_state_dict(best_model[0])
    classifier.load_state_dict(best_model[1])

    gender_results = validate_by_gender(val_loader, model, classifier, criterion, args)

    log_dict = {
        "BestEpoch": epoch,
        "Accuracy": best_acc[0],
        "Precision": best_acc[1],
        "Recall": best_acc[2],
        "F1": best_acc[3],
        "Gender": gender_results
    }
    
    print('{} finished'.format(args.model_name))

    update_json(args.model_name, log_dict, path=os.path.join(args.save_dir, args.save_name))
    print(f"✅ Final results saved with update_json at {args.save_name}")
    print("=====================================================\n")
    
if __name__ == '__main__':
    main()
