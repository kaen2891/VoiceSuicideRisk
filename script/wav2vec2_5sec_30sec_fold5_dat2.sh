
MODEL="facebook/wav2vec2-base"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="bs32_lr5e-5_ep50_seed${s}_fold5_30sec_all_dat2"
        CUDA_VISIBLE_DEVICES=1 python main.py --tag $TAG \
                                        --dataset psychiatry \
                                        --seed $s \
                                        --class_split psychiatry \
                                        --n_cls 2 \
                                        --epochs 50 \
                                        --annotation /home/jovyan/speech/dataset/fold_split_meta/fold_5.csv \
                                        --task_group all \
                                        --domain_adaptation2 \
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
                                        --save_name wav2vec2 \
                                        --print_freq 100

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

    done
done
