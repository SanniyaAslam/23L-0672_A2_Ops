#!/usr/bin/env bash
# Task 3.5 - 6+ runs, each differing from the baseline by ONE meaningful change
set -e
cd "$(dirname "$0")"
python train.py --model cnn      --run-name 01_cnn_scratch
python train.py --model transfer --run-name 02_mnv2_lr1e-3
python train.py --model transfer --run-name 03_mnv2_lr1e-4  --lr 1e-4
python train.py --model transfer --run-name 04_mnv2_no_aug  --no-aug
python train.py --model transfer --run-name 05_mnv2_frozen  --no-finetune
python train.py --model transfer --run-name 06_mnv2_img160  --img-size 160
python train.py --model transfer --run-name 07_mnv2_autolog --autolog
