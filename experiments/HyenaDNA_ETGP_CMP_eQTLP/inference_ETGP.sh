#!/bin/bash

python -m scripts/inference_ETGP.py wandb=null experiment=hg38/enhancer_promoter_benchmark \
train.pretrained_model_path=ETGP/checkpoints/val/loss.ckpt \
train.test=True