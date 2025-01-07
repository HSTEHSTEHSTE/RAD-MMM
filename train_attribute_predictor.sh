#!/usr/bin/env bash

source /home/hltcoe/xli/.bashrc
source /home/hltcoe/xli/anaconda3/etc/profile.d/conda.sh

# export CUDA_VISIBLE_DEVICES=$(free-gpu)
echo $CUDA_VISIBLE_DEVICES

conda activate lightning
cd /home/hltcoe/xli/ARTS/RAD-MMM

# orig
# python tts_main.py fit -c configs/RADMMM_f0model_config.yaml -c configs/RADMMM_energymodel_config.yaml -c configs/RADMMM_durationmodel_config.yaml -c configs/RADMMM_vpredmodel_config.yaml -c configs/RADMMM_train_config.yaml -c configs/RADMMM_opensource_data_config_phonemizerless.yaml -c configs/RADMMM_model_config.yaml --trainer.num_nodes=1 --trainer.devices=1 --trainer.logger=WandbLogger 

# # knnvc augmentation: voxceleb
# python tts_main.py fit -c configs/RADMMM_f0model_config.yaml -c configs/RADMMM_energymodel_config.yaml -c configs/RADMMM_durationmodel_config.yaml -c configs/RADMMM_vpredmodel_config.yaml -c configs/RADMMM_train_config.yaml -c configs/RADMMM_opensource_data_config_phonemizerless_knnvc.yaml -c configs/RADMMM_model_config.yaml --trainer.num_nodes=1 --trainer.devices=1 --trainer.logger=WandbLogger --ckpt_path /home/hltcoe/xli/ARTS/RAD-MMM/exp/decoder-k/latest-epoch_4-iter_69999.ckpt

# knnvc augmentation: libri
python tts_main.py fit -c configs/RADMMM_f0model_config.yaml -c configs/RADMMM_energymodel_config.yaml -c configs/RADMMM_durationmodel_config.yaml -c configs/RADMMM_vpredmodel_config.yaml -c configs/RADMMM_train_config.yaml -c configs/RADMMM_opensource_data_config_phonemizerless_knnvc_libri.yaml -c configs/RADMMM_model_config.yaml --trainer.num_nodes=1 --trainer.devices=1 --trainer.logger=WandbLogger

# --ckpt_path /home/hltcoe/xli/ARTS/RAD-MMM/exp/decoder-a/latest-epoch_20-iter_124999.ckpt