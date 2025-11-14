import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ---------------------------
# 1️⃣ 모든 음성 파일 경로 + 레이블 수집
# ---------------------------
base_dir = './speech_cut_ver2'
tasks = ['Task1', 'Task2']
labels = ['HR', 'LR']

def get_task_group(task_name, wav_path):
    """
    파일명 끝의 숫자를 보고 Incongruent / Color / Word 중 어떤 그룹인지 결정
    """
    try:
        # 예: subject_004_1_4.wav → 마지막 번호 추출 (4)
        num = int(os.path.basename(wav_path).split('_')[-1].split('.')[0])
    except Exception:
        return 'Unknown'

    if task_name == 'Task1':
        if num in [3, 5]:
            return 'Incongruent'
        elif num in [2, 6]:
            return 'Color'
        elif num in [1, 4]:
            return 'Word'
    elif task_name == 'Task2':
        if num in [1, 4]:
            return 'Incongruent'
        elif num in [3, 5]:
            return 'Color'
        elif num in [2, 6]:
            return 'Word'
    return 'Unknown'

data = []

for task in tasks:
    for label in labels:
        path = os.path.join(base_dir, task, label)
        if not os.path.exists(path):
            continue
        subjects = sorted(os.listdir(path))
        for subj in subjects:
            subj_path = os.path.join(path, subj)
            if not os.path.isdir(subj_path):
                continue
            wavs = glob.glob(os.path.join(subj_path, '*.wav'))
            for wav in wavs:
                task_group = get_task_group(task, wav)
                data.append({
                    'task': task,
                    'task_group': task_group,   # ✅ 새 필드 추가
                    'label': 1 if label == 'HR' else 0,
                    'subject_id': subj,
                    'wav_path': wav
                })

df = pd.DataFrame(data)
print(f"✅ Total wav files: {len(df)}")
print(f"✅ Unique subjects: {df['subject_id'].nunique()}")
print(df[['task', 'task_group', 'label', 'subject_id', 'wav_path']].head())

# ---------------------------
# 2️⃣ Stratified 5-Fold split (subject-level)
# ---------------------------
subjects = df[['subject_id', 'label']].drop_duplicates()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

os.makedirs('./fold_split_full_task_cut_ver2', exist_ok=True)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(subjects['subject_id'], subjects['label'])):
    train_subj = subjects.iloc[train_idx]['subject_id'].tolist()
    test_subj = subjects.iloc[test_idx]['subject_id'].tolist()

    # subject 기준으로 join
    train_df = df[df['subject_id'].isin(train_subj)].copy()
    test_df = df[df['subject_id'].isin(test_subj)].copy()

    train_df['set'] = 'train'
    test_df['set'] = 'test'

    fold_df = pd.concat([train_df, test_df], ignore_index=True)
    save_path = f'./fold_split_full_task_cut_ver2/fold_{fold_idx + 1}.csv'
    fold_df.to_csv(save_path, index=False)

    print(f"[Fold {fold_idx + 1}] Train files: {len(train_df)}, Test files: {len(test_df)}")

print("\n🎯 All fold splits saved to ./fold_split_full_task_cut_ver2/")