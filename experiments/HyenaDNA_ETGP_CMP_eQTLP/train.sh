#!/bin/bash

python -m train ++model.checkpoint_mixer=True ++model.checkpoint_mlp=True wandb=null \
experiment=hg38/hg38_hyena \
train.pretrained_model_path=pretrained_models/weights.ckpt \
callbacks.model_checkpoint.every_n_train_steps=3000 \
model.n_layer=8 \
model.d_model=256 \
dataset.batch_size=1 \
train.global_batch_size=1 \
dataset.max_length=450000 \
optimizer.lr=6e-4 \
trainer.devices=1

#
