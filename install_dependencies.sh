#!/bin/bash
# Install PyTorch and Python Packages

# conda create -n pymarl python=3.8 -y
# conda activate pymarl
 
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia pymongo -y
pip install protobuf==3.19.5 sacred==0.7.5 numpy scipy gymnasium==1.0.0 matplotlib seaborn \
    pyyaml==5.3.1 pygame pytest probscale imageio snakeviz tensorboard-logger lbforaging==2.0.0
pip install git+https://github.com/oxwhirl/smac.git@26f4c4e4d1ebeaf42ecc2d0af32fac0774ccc678

# you need to move EDMAC/smac_maps/* to StarCraftII/Maps/SMAC_Maps after SC2 was installed