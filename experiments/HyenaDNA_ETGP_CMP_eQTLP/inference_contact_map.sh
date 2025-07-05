#!/bin/bash

python -m scripts/inference_contact_map.py wandb=null experiment=hg38/akita_benchmark \
train.pretrained_model_path=contact_map/checkpoints/val/loss.ckpt \
train.test=True