from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset
from copy import deepcopy

from .util import get_individual_samples_torchaudio, cut_pad_sample_torchaudio
from transformers import AutoFeatureExtractor
from .util import get_individual_samples_torchaudio, cut_pad_sample_torchaudio
from transformers import AutoFeatureExtractor

class PsychiatryDataset(Dataset):
    def __init__(self, train_flag, args, annotation, print_flag=True, mean_std=False):
        """
        PsychiatryDataset
        - 성별(gender) 정보를 segment 단위로 확장하여 저장
        - wav 파일 1개에서 여러 샘플(segment)이 추출될 경우 gender도 동일하게 복제됨
        """
        self.train_flag = train_flag
        self.args = args
        self.mean_std = mean_std
        self.data_folder = args.data_folder
        self.sample_rate = args.sample_rate
        self.model_name = args.model
        self.task_group = getattr(args, "task_group", "all")

        # ======================================================
        # 1️⃣ Annotation 파일 읽기
        # ======================================================
        df = pd.read_csv(args.annotation)
        split = "train" if train_flag else "test"
        df = df[df["set"] == split].reset_index(drop=True)

        # ✅ task_group filtering
        if "task_group" in df.columns and self.task_group.lower() != "all":
            df = df[df["task_group"].str.lower() == self.task_group.lower()]
            if print_flag:
                print(f"⚙️ Filtering by task_group = {self.task_group} → {len(df)} samples")

        self.data = df.reset_index(drop=True)
        samples = self.data["wav_path"].tolist()
        labels = self.data["label"].tolist()
        genders = self.data["sex"].tolist()

        if print_flag:
            print("=" * 30)
            print(f"Building PsychiatryDataset ({split})")
            print(f"Model: {self.model_name}")
            print(f"Task Group: {self.task_group}")
            print(f"Total audio files: {len(samples)}")

        # ======================================================
        # 2️⃣ torchaudio 기반 individual sample 생성
        # ======================================================
        self.suc_list = []
        self.gender_list_expanded = []  # ✅ segment 단위 gender 확장 리스트

        for wav_path, label, gender in tqdm(zip(samples, labels, genders),
                                            total=len(samples),
                                            desc="Extracting samples"):
            sample_data = get_individual_samples_torchaudio(
                args,
                os.path.join(self.data_folder, wav_path),
                self.sample_rate,
                args.n_cls,
                self.train_flag,
                label
            )
            self.suc_list.extend([(data[0], data[1]) for data in sample_data])
            self.gender_list_expanded.extend([gender] * len(sample_data))  # ✅ gender 복제

        assert len(self.suc_list) == len(self.gender_list_expanded), \
            f"Gender list length mismatch: {len(self.gender_list_expanded)} vs {len(self.suc_list)}"

        # ======================================================
        # 3️⃣ Transformer Feature Extractor 로드
        # ======================================================
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)

        # ======================================================
        # 4️⃣ 오디오 → feature 변환
        # ======================================================
        self.audio_images = []
        for idx in tqdm(range(len(self.suc_list)), desc="Extracting transformer features"):
            audio, label = self.suc_list[idx]
            gender = self.gender_list_expanded[idx]

            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding="longest"
            )

            feature_tensor = (
                torch.from_numpy(inputs["input_values"][0].numpy())
                if isinstance(inputs["input_values"], np.ndarray)
                else inputs["input_values"][0]
            )
            self.audio_images.append((feature_tensor, label, gender))  # ✅ gender 함께 저장

        # ======================================================
        # 5️⃣ class 분포 출력
        # ======================================================
        self.class_nums = np.zeros(args.n_cls)
        for _, label, _ in self.audio_images:
            self.class_nums[label] += 1
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100

        if print_flag:
            print("=" * 30)
            print("Label distribution:")
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print(f"Class {i} ({args.cls_list[i]:<10}): {int(n)} ({p:.1f}%)")

    # ======================================================
    # 6️⃣ Dataset 기본 메소드
    # ======================================================
    def __getitem__(self, index):
        audio_feature, label, gender = self.audio_images[index]
        return audio_feature.squeeze(0), label, gender

    def __len__(self):
        return len(self.audio_images)