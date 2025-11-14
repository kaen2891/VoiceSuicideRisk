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

from curses import meta
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc as calc_auc
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, confusion_matrix

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
    if args.eval:
        args.save_folder = os.path.join(args.save_dir, 'eval', args.model_name)
    else:
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

###############################################################
# Data Loader
###############################################################
def set_loader(args):
    if args.eval:
        val_dataset = PsychiatryDataset(train_flag=False, args=args, annotation=args.annotation, print_flag=True)
        val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
        )

        return None, val_loader, args
    
    else:
    
        train_dataset = PsychiatryDataset(train_flag=True, args=args, annotation=args.annotation, print_flag=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
        val_dataset = PsychiatryDataset(train_flag=False, args=args, annotation=args.annotation, print_flag=True)
        val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
        )
        
    
        return train_loader, val_loader, args


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
    
    if args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']
        # HOTFIX: always use dataparallel during SSL pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            if not 'mlp_head' in k: #del mlp_head
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        
        if ckpt.get('classifier', None) is not None:
            classifier.load_state_dict(ckpt['classifier'], strict=True)

        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))



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

def set_seed(seed): # for reproducibility
    import torch, numpy as np, random, os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

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

def validate(val_loader, model, classifier, criterion, args,
             best_acc, best_model=None):

    model.eval()
    classifier.eval()

    gender_map = {'M': 0, 'm': 0, 'F': 1, 'f': 1}

    all_preds = []
    all_labels = []
    all_probs = []
    all_genders = []

    with torch.no_grad():
        for images, labels, genders in val_loader:
            images = images.cuda()
            labels = labels.cuda()

            with torch.cuda.amp.autocast():
                images = torch.squeeze(images, 1)
                outputs = model(images)
                hidden = outputs.last_hidden_state.mean(dim=1)
                logits = classifier(hidden)

            preds = torch.argmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)[:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # 🔥 gender 통일
            genders_clean = []
            for g in genders:
                if isinstance(g, str):
                    genders_clean.append(gender_map[g])
                else:
                    genders_clean.append(int(g))

            all_genders.extend(genders_clean)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_genders = np.array(all_genders)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    if f1 > best_acc[-1]:
        best_acc = [acc, precision, recall, f1, auc]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]
        save_bool = True

    return best_acc, best_model, save_bool, cm, auc, all_labels, all_preds, all_probs, all_genders



###############################################################
# Gender-wise
###############################################################
def validate_by_gender(all_labels, all_preds, all_probs, all_genders, args):

    results = {}
    gender_groups = np.unique(all_genders)

    for gender in gender_groups:
        idxs = np.where(all_genders == gender)[0]

        y_true = all_labels[idxs]
        y_pred = all_preds[idxs]
        y_prob = all_probs[idxs]

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = None

        results[str(gender)] = {
            "Samples": len(y_true),
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc
        }

    return results

def extract_embeddings(data_loader, model, classifier, device="cuda"):
    model.eval()
    classifier.eval()

    embeddings = []
    labels = []
    genders = []

    gender_map = {'M': 0, 'm': 0, 'F': 1, 'f': 1}

    with torch.no_grad():
        for images, y, g in data_loader:
            images = images.to(device)
            y = y.to(device)

            images = torch.squeeze(images, 1)
            outputs = model(images)
            hidden = outputs.last_hidden_state.mean(dim=1)

            # embedding 저장
            embeddings.append(hidden.cpu().numpy())
            labels.append(y.cpu().numpy())

            # gender 저장
            g_clean = []
            for item in g:
                if isinstance(item, str):
                    g_clean.append(gender_map[item])
                else:
                    g_clean.append(int(item))
            genders.append(np.array(g_clean))

    embeddings = np.vstack(embeddings)     # [N, 768]
    labels = np.hstack(labels)             # [N]
    genders = np.hstack(genders)           # [N]

    return embeddings, labels, genders

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_tsne_paper_style(embeddings, targets, class_names, save_path, title="t-SNE"):
    
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        init='pca',
        random_state=42
    )
    emb_2d = tsne.fit_transform(embeddings)

    # 색상 (필요한 만큼 확장 가능)
    color_map = {
        0: "#E64B35FF",  # Red-ish
        1: "#4DBBD5FF",  # Blue-ish
        2: "#00A087FF",  # Green-ish
        3: "#3C5488FF",  # Navy
        4: "#F39B7FFF"
    }

    plt.figure(figsize=(8, 7))

    for cls in np.unique(targets):
        idxs = np.where(targets == cls)[0]

        plt.scatter(
            emb_2d[idxs, 0],
            emb_2d[idxs, 1],
            s=18,
            alpha=0.75,
            linewidth=0.3,
            edgecolors='black',
            color=color_map.get(cls, '#333333'),
            label=class_names[cls]
        )

    # === 축 이름 ===
    plt.xlabel("comp-1", fontsize=14)
    plt.ylabel("comp-2", fontsize=14)

    # === 제목 ===
    plt.title(title, fontsize=16)

    # === grid ===
    plt.grid(True, linestyle='--', alpha=0.4)

    # === legend 박스 ===
    plt.legend(
        frameon=True,
        facecolor='white',
        edgecolor='black',
        fontsize=12,
        loc='upper right'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[SAVED] {save_path}")

###############################################################
# MAIN
###############################################################
def main():

    args = parse_args()
    set_seed(args.seed)

    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

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
    


    scaler = torch.cuda.amp.GradScaler()

    best_acc = [0, 0, 0, 0, 0]  
    best_model = None

    ###############################################################
    # TRAIN
    ###############################################################
    if not args.eval:
        print("Start training...")

        for epoch in range(1, args.epochs + 1):

            adjust_learning_rate(args, optimizer, epoch)

            loss, acc = train(
                train_loader, model, classifier, domain_classifier,
                criterion, optimizer, epoch, args, scaler
            )

            best_acc, best_model, save_bool, cm, auc, all_labels, all_preds, all_probs, all_genders = validate(
                val_loader, model, classifier, criterion, args, best_acc, best_model
            )

            if save_bool:
                save_path = os.path.join(args.save_folder, f"best_epoch_{epoch}.pth")
                save_model(model, optimizer, args, epoch, save_path, classifier)
                print(f"Best model saved at epoch {epoch}")

            if epoch % args.save_freq == 0:
                save_path = os.path.join(args.save_folder, f"epoch_{epoch}.pth")
                save_model(model, optimizer, args, epoch, save_path, classifier)

        # Load best model
        model.load_state_dict(best_model[0])
        classifier.load_state_dict(best_model[1])

    else:
        best_acc, best_model, save_bool, cm, auc, all_labels, all_preds, all_probs, all_genders = validate(
            val_loader, model, classifier, criterion, args, best_acc, best_model
        )

    ###############################################################
    # Save ROC Curve (Option 2)
    ###############################################################
    roc_path = os.path.join(args.save_folder, "best_model_roc.png")
    print('roc_path', roc_path)

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = calc_auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", lw=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Best Model)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"[ROC SAVED] {roc_path}")

    ###############################################################
    # Save Confusion Matrix (Option 3)
    ###############################################################
    cm_path = os.path.join(args.save_folder, "confusion_matrix.png")
    print('cm_path', cm_path)

    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=args.cls_list)
    disp.plot(cmap="Blues", values_format='d', ax=plt.gca(), colorbar=False)
    plt.title("Confusion Matrix (Best Model)")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"[CONFUSION MATRIX SAVED] {cm_path}")

    gender_results = validate_by_gender(all_labels, all_preds, all_probs, all_genders, args)


    gender_groups = np.unique(all_genders)

    gender_label_map = {0: "Male", 1: "Female"}
    
    for gender in gender_groups:

        idxs = np.where(all_genders == gender)[0]
        y_true = all_labels[idxs]
        y_prob = all_probs[idxs]
        y_pred = np.array(all_preds[idxs])

        gender_name = gender_label_map.get(gender, str(gender))   # 숫자를 문자열로 변환

        # Gender-wise ROC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            gender_auc = calc_auc(fpr, tpr)

            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, lw=2, label=f"AUC = {gender_auc:.2f}")
            plt.plot([0, 1], [0, 1], 'k--')

            plt.title(f"ROC ({gender_name})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(args.save_folder, f"roc_{gender_name}.png"), dpi=300)
            plt.close()

        except:
            print(f"[SKIP] ROC not available for {gender_name}")

        # Gender-wise Confusion Matrix
        cm_g = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_g, display_labels=args.cls_list)
        disp.plot(cmap="Blues", values_format='d', ax=plt.gca(), colorbar=False)
        plt.title(f"Confusion Matrix ({gender_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_folder, f"cm_{gender_name}.png"), dpi=300)
        plt.close()
    print('save_folder', args.save_folder)

    ###############################################################
    # Final Save Log
    ###############################################################
    log_dict = {
        "Accuracy": best_acc[0],
        "Precision": best_acc[1],
        "Recall": best_acc[2],
        "F1": best_acc[3],
        "AUC": best_acc[4],
        "Gender": gender_results
    }

    update_json(args.model_name, log_dict, path=os.path.join(args.save_dir, args.save_name))

    embeddings, tsne_labels, tsne_genders = extract_embeddings(val_loader, model, classifier)

    # label 기준 t-SNE
    label_names = {0: "low", 1: "high"}
    plot_tsne_paper_style(
        embeddings,
        tsne_labels,
        label_names,
        save_path=os.path.join(args.save_folder, "tsne_label.png"),
        title="t-SNE by Label"
    )

    # gender 기준 t-SNE
    gender_names = {0: "Male", 1: "Female"}
    plot_tsne_paper_style(
        embeddings,
        tsne_genders,
        gender_names,
        save_path=os.path.join(args.save_folder, "tsne_gender.png"),
        title="t-SNE by Gender"
    )




    print("\n========== Training Finished ==========\n")
    print(args.model_name)

    
if __name__ == '__main__':
    main()
