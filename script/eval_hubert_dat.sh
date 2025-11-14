#!/bin/bash
MODEL="facebook/hubert-base-ls960"
SEED="1"

# fold 리스트
FOLDS="1 2 3 4 5"

# task group 리스트
#TASKS=("Color" "Word" "Incongruent" "all")
TASKS=("all")

for s in $SEED
do
    for m in $MODEL
    do
        for f in $FOLDS
        do
            for t in "${TASKS[@]}"
            do
                # tag 이름을 fold, task 포함해서 만들기
                if [ "$t" == "" ]; then
                    TAG="eval_fold${f}_none"
                else
                    TAG="eval_fold${f}_${t,,}_dat"   # 소문자 변환
                fi

                echo "Running fold=$f, task=$t"

                CUDA_VISIBLE_DEVICES=1 python main.py \
                    --tag $TAG \
                    --dataset psychiatry \
                    --seed $s \
                    --class_split psychiatry \
                    --n_cls 2 \
                    --epochs 50 \
                    --annotation /home/jovyan/speech/dataset/fold_split_meta/fold_${f}.csv \
                    --task_group "$t" \
                    --batch_size 32 \
                    --optimizer adam \
                    --learning_rate 5e-5 \
                    --weight_decay 1e-6 \
                    --cosine \
                    --model $m \
                    --desired_length 5 \
                    --divide_length 5 \
                    --ma_update \
                    --ma_beta 0.5 \
                    --method ce \
                    --save_name hubert \
                    --eval \
                    --pretrained_ckpt ./save/psychiatry_facebook/hubert-base-ls960_ce_bs32_lr5e-5_ep50_seed1_fold${f}_30sec_dat/best.pth
            done
        done
    done
done