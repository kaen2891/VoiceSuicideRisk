from collections import namedtuple
import os
import math
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torchaudio
from torchaudio import transforms as T

__all__ = ['get_mean_and_std', 'get_individual_samples_torchaudio', 'get_score']


# ===============================================================
    

def get_mean_and_std(dataset):
    """ Compute the mean and std value of mel-spectrogram """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

    cnt = 0
    fst_moment = torch.zeros(1)
    snd_moment = torch.zeros(1)
    for inputs, _, _ in dataloader:
        b, c, h, w = inputs.shape
        nb_pixels = b * h * w

        fst_moment += torch.sum(inputs, dim=[0,2,3])
        snd_moment += torch.sum(inputs**2, dim=[0,2,3])
        cnt += nb_pixels

    mean = fst_moment / cnt
    std = torch.sqrt(snd_moment/cnt - mean**2)

    return mean, std
# ==========================================================================


# ==========================================================================
""" data preprocessing """

def cut_pad_sample_torchaudio(data, train_flag, args):
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    if train_flag:
        target_duration = args.desired_length * args.sample_rate
    else:
        target_duration = 30 * args.sample_rate

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
        if data.dim() == 1:
            data = data.unsqueeze(0)
    else:
        ratio = math.ceil(target_duration / data.shape[-1])
        data = data.repeat(1, ratio)
        data = data[..., :target_duration]
        data = fade_out(data)
    
    return data

def get_individual_samples_torchaudio(args, wav_path, sample_rate, n_cls, train_flag, label):
    sample_data = []    

    data, sr = torchaudio.load(wav_path)
        
    if data.size(0) == 2: # if stereo 
        data = torch.mean(data, dim=0).unsqueeze(0)
        
    if sr != sample_rate:
        resample = T.Resample(sr, sample_rate)
        data = resample(data)
    
    fade_samples_ratio = 16
    fade_samples = int(sample_rate / fade_samples_ratio)
    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
    data = fade(data)
        
    if train_flag:
        divide_count = 30 // args.divide_length
        for i in range(divide_count):
            start = i * args.divide_length * sample_rate
            end = (i+1) * args.divide_length * sample_rate
            divided_wav = data[:, i*args.divide_length*sample_rate:(i+1)*args.divide_length*sample_rate]
            sample_data.append((divided_wav, label))
    else:
        if args.cut_test:
            divide_count = 30 // args.divide_length
            for i in range(divide_count):
                start = i * args.divide_length * sample_rate
                end = (i+1) * args.divide_length * sample_rate
                divided_wav = data[:, i*args.divide_length*sample_rate:(i+1)*args.divide_length*sample_rate]
                sample_data.append((divided_wav, label))
        else:
            sample_data.append((data, label))
            return sample_data
    
    padded_sample_data = []
    for data, label in sample_data:
        data = cut_pad_sample_torchaudio(data, train_flag, args) # --> resample to [1, 80000] --> 5 seconds
        padded_sample_data.append((data, label))
    return padded_sample_data



# ==========================================================================


# ==========================================================================
""" evaluation metric """
def get_score(hits, counts, pflag=False):
    # normal accuracy
    print(hits)
    print(counts)
    sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    sc = (sp + se) / 2.0

    if pflag:
        # print("************* Metrics ******************")
        print("S_p: {}, S_e: {}, Score: {}".format(sp, se, sc))

    return sp, se, sc
# ==========================================================================
