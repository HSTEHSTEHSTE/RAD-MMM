#!/usr/bin/env bash

source /home/hltcoe/xli/.bashrc
source /home/hltcoe/xli/anaconda3/etc/profile.d/conda.sh

# export CUDA_VISIBLE_DEVICES=$(free-gpu)
echo $CUDA_VISIBLE_DEVICES

conda activate lightning
cd /home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024
source env.sh
cd ../RAD-MMM
python knnvc_augment.py