#!/bin/bash

# 实验相关执行命令，包含调参

mkdir -p log

## baseline

CUDA_VISIBLE_DEVICES=0 python run.py --model TextCNN > log/1.log &

# no dropout
CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --no_dropout 1 > log/2.log &


## FGSM
# 参数范围
# --adv_epsilon 0.1 0.05 0.01 0.005 0.001

CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial fgsm --adv_epsilon 0.1 > log/3.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial fgsm --adv_epsilon 0.05 > log/4.log &
CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial fgsm --adv_epsilon 0.01 > log/5.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial fgsm --adv_epsilon 0.005 > log/6.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial fgsm --adv_epsilon 0.001 > log/7.log &

# no dropout
CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --adversarial fgsm --adv_epsilon 0.1 --no_dropout 1 > log/8.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial fgsm --adv_epsilon 0.05 --no_dropout 1 > log/9.log &
CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial fgsm --adv_epsilon 0.01 --no_dropout 1 > log/10.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial fgsm --adv_epsilon 0.005 --no_dropout 1 > log/11.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial fgsm --adv_epsilon 0.001 --no_dropout 1 > log/12.log &


## PGD
# 参数范围
# --adv_epsilon 0.05 0.01 0.005
# --adv_alpha  0.1 0.01 0.001
# --adv_k 2 4 8

CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.1 --adv_k 2 > log/13.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.1 --adv_k 4 > log/14.log &
CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.1 --adv_k 8 > log/15.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.01 --adv_k 2 > log/16.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.01 --adv_k 4 > log/17.log &
CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.01 --adv_k 8 > log/18.log &

CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.001 --adv_k 2 > log/19.log &
CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.001 --adv_k 4 > log/20.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.001 --adv_k 8 > log/21.log &
CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.1 --adv_k 2 > log/22.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.1 --adv_k 4 > log/23.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.1 --adv_k 8 > log/24.log &

CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.01 --adv_k 2 > log/25.log &
CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.01 --adv_k 4 > log/26.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.01 --adv_k 8 > log/27.log &
CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.001 --adv_k 2 > log/28.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.001 --adv_k 4 > log/29.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.001 --adv_k 8 > log/30.log &

CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.1 --adv_k 2 > log/31.log &
CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.1 --adv_k 4 > log/32.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.1 --adv_k 8 > log/33.log &
CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.01 --adv_k 2 > log/34.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.01 --adv_k 4 > log/35.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.01 --adv_k 8 > log/36.log &

CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.001 --adv_k 2 > log/37.log &
CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.001 --adv_k 4 > log/38.log &
CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.001 --adv_k 8 > log/39.log &

# no dropout
CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.1 --adv_k 2 --no_dropout 1 > log/40.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.1 --adv_k 4 --no_dropout 1 > log/41.log &
CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.1 --adv_k 8 --no_dropout 1 > log/42.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.01 --adv_k 2 --no_dropout 1 > log/43.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.01 --adv_k 4 --no_dropout 1 > log/44.log &
CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.01 --adv_k 8 --no_dropout 1 > log/45.log &
CUDA_VISIBLE_DEVICES=0 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.001 --adv_k 2 --no_dropout 1 > log/46.log &

CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.001 --adv_k 4 --no_dropout 1 > log/47.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.05 --adv_alpha 0.001 --adv_k 8 --no_dropout 1 > log/48.log &
CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.1 --adv_k 2 --no_dropout 1 > log/49.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.1 --adv_k 4 --no_dropout 1 > log/50.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.1 --adv_k 8 --no_dropout 1 > log/51.log &
CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.01 --adv_k 2 --no_dropout 1 > log/52.log &
CUDA_VISIBLE_DEVICES=0 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.01 --adv_k 4 --no_dropout 1 > log/53.log &

CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.01 --adv_k 8 --no_dropout 1 > log/54.log &
CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.001 --adv_k 2 --no_dropout 1 > log/55.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.001 --adv_k 4 --no_dropout 1 > log/56.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.01 --adv_alpha 0.001 --adv_k 8 --no_dropout 1 > log/57.log &
CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.1 --adv_k 2 --no_dropout 1 > log/58.log &
CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.1 --adv_k 4 --no_dropout 1 > log/59.log &
CUDA_VISIBLE_DEVICES=0 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.1 --adv_k 8 --no_dropout 1 > log/60.log &

CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.01 --adv_k 2 --no_dropout 1 > log/61.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.01 --adv_k 4 --no_dropout 1 > log/62.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.01 --adv_k 8 --no_dropout 1 > log/63.log &
CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.001 --adv_k 2 --no_dropout 1 > log/64.log &
CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.001 --adv_k 4 --no_dropout 1 > log/65.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial pgd --adv_epsilon 0.005 --adv_alpha 0.001 --adv_k 8 --no_dropout 1 > log/66.log &


## FreeAT
# 参数范围
# --adv_epsilon 0.05 0.01 0.005 0.001
# --adv_k 2 4 8

CUDA_VISIBLE_DEVICES=0 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.05 --adv_k 2 > log/67.log &
CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.05 --adv_k 4 > log/68.log &
CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.05 --adv_k 8 > log/69.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.01 --adv_k 2 > log/70.log &
CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.01 --adv_k 4 > log/71.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.01 --adv_k 8 > log/72.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.005 --adv_k 2 > log/73.log &

CUDA_VISIBLE_DEVICES=0 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.005 --adv_k 4 > log/74.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.005 --adv_k 8 > log/75.log &
CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.001 --adv_k 2 > log/76.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.001 --adv_k 4 > log/77.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.001 --adv_k 8 > log/78.log &

# no dropout
CUDA_VISIBLE_DEVICES=0 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.05 --adv_k 2 --no_dropout 1 > log/79.log &
CUDA_VISIBLE_DEVICES=1 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.05 --adv_k 4 --no_dropout 1 > log/80.log &
CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.05 --adv_k 8 --no_dropout 1 > log/81.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.01 --adv_k 2 --no_dropout 1 > log/82.log &
CUDA_VISIBLE_DEVICES=5 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.01 --adv_k 4 --no_dropout 1 > log/83.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.01 --adv_k 8 --no_dropout 1 > log/84.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.005 --adv_k 2 --no_dropout 1 > log/85.log &

CUDA_VISIBLE_DEVICES=0 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.005 --adv_k 4 --no_dropout 1 > log/86.log &
CUDA_VISIBLE_DEVICES=6 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.005 --adv_k 8 --no_dropout 1 > log/87.log &
CUDA_VISIBLE_DEVICES=3 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.001 --adv_k 2 --no_dropout 1 > log/88.log &
CUDA_VISIBLE_DEVICES=4 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.001 --adv_k 4 --no_dropout 1 > log/89.log &
CUDA_VISIBLE_DEVICES=7 python run.py --model TextCNN --adversarial freeat --adv_epsilon 0.001 --adv_k 8 --no_dropout 1 > log/90.log &
