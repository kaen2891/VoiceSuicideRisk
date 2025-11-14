import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ---------------------------
# 1️⃣ 메타데이터 로드 (sex, age, 심리척도)
# ---------------------------
meta_path = './voice_list.csv'  # ← 파일 경로 확인
meta_df = pd.read_csv(meta_path, sep='\t')

# NaN 제거 후 num을 int로 변환
meta_df = meta_df.dropna(subset=['num'])
meta_df['num'] = meta_df['num'].astype(int)

print(f"✅ Metadata loaded: {len(meta_df)} subjects")
print(meta_df.head())

# ---------------------------
# 2️⃣ 모든 wav 파일 및 라벨 수집
# ---------------------------
base_dir = './speech'
tasks = ['Task1', 'Task2']
labels = ['HR', 'LR']

def get_task_group(task_name, wav_path):
    """파일명 끝의 숫자를 보고 Incongruent / Color / Word 그룹 결정"""
    try:
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
                # subject 폴더 이름에서 숫자만 추출 (예: subject_004 → 4)
                subj_num = int(''.join(filter(str.isdigit, subj))) if any(ch.isdigit() for ch in subj) else None
                data.append({
                    'task': task,
                    'task_group': task_group,
                    'label': 1 if label == 'HR' else 0,
                    'subject_id': subj,
                    'num': subj_num,
                    'wav_path': wav
                })

df = pd.DataFrame(data)
print(f"✅ Total wav files: {len(df)}")
print(f"✅ Unique subjects: {df['subject_id'].nunique()}")

# ---------------------------
# 3️⃣ 메타데이터 병합
# ---------------------------
merged_df = df.merge(meta_df, on='num', how='left')

print("✅ Merged dataframe sample:")
print(merged_df.head(3))

# ---------------------------
# 4️⃣ Stratified 5-Fold Split (subject-level)
# ---------------------------
subjects = merged_df[['subject_id', 'label_x']].drop_duplicates()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

os.makedirs('./fold_split_meta', exist_ok=True)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(subjects['subject_id'], subjects['label_x'])):
    train_subj = subjects.iloc[train_idx]['subject_id'].tolist()
    test_subj = subjects.iloc[test_idx]['subject_id'].tolist()

    train_df = merged_df[merged_df['subject_id'].isin(train_subj)].copy()
    test_df = merged_df[merged_df['subject_id'].isin(test_subj)].copy()

    train_df['set'] = 'train'
    test_df['set'] = 'test'

    fold_df = pd.concat([train_df, test_df], ignore_index=True)
    save_path = f'./fold_split_meta/fold_{fold_idx + 1}.csv'
    fold_df.to_csv(save_path, index=False)

    print(f"[Fold {fold_idx + 1}] Train: {len(train_df)} | Test: {len(test_df)} saved to {save_path}")

print("\n🎯 All folds with metadata saved to ./fold_split_meta/")