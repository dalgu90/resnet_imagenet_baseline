#!/bin/sh
export CUDA_VISIBLE_DEVICES=1

checkpoint_dir="./$1"
test_output="$checkpoint_dir/eval_test.txt"
train_output="$checkpoint_dir/eval_train.txt"
batch_size=32
gpu_fraction=0.96

python eval.py --checkpoint_dir $checkpoint_dir \
               --output $test_output \
               --batch_size $batch_size \
               --gpu_fraction $gpu_fraction

python eval.py --checkpoint_dir $checkpoint_dir \
               --output $train_output \
               --batch_size $batch_size \
               --gpu_fraction $gpu_fraction \
               --train_data True
