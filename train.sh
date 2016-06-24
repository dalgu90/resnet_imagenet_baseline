#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
train_dir=./train1

python train.py --train_dir $train_dir \
    --batch_size 32 \
    --max_steps 25000 \
    --initial_lr 0.001 \
    --lr_step_epoch 2.0 \
    --lr_decay 0.1 \
    --l2_weight 0.0001 \
    --momentum 0.9 \
    --gpu_fraction 0.96 \
    --checkpoint_interval 1000 \
