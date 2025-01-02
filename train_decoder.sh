#!/usr/bin/env bash

source /home/hltcoe/xli/.bashrc
source /home/hltcoe/xli/anaconda3/etc/profile.d/conda.sh

# export CUDA_VISIBLE_DEVICES=$(free-gpu)
echo $CUDA_VISIBLE_DEVICES

conda activate lightning
cd /home/hltcoe/xli/ARTS/RAD-MMM
# python tts_main.py fit -c configs/RADMMM_train_config.yaml -c configs/RADMMM_opensource_data_config_phonemizerless.yaml -c configs/RADMMM_model_config.yaml --trainer.num_nodes=1 --trainer.devices=1 --trainer.logger=WandbLogger
python tts_main.py fit -c configs/RADMMM_train_config.yaml -c configs/RADMMM_opensource_data_config_phonemizerless.yaml -c configs/RADMMM_model_config.yaml --trainer.num_nodes=1 --trainer.devices=1 --trainer.logger=WandbLogger --ckpt_path /home/hltcoe/xli/ARTS/RAD-MMM/exp/decoder-k/latest-epoch_23-iter_139999.ckpt