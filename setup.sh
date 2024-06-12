#!/bin/bash

conda create -n dnalongbench python=3.8 
conda activate dnalongbench

pip install tensorflow==2.4.1
pip install numpy==1.19.2

pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 torchtext==0.16.0 -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install torchmetrics kipoiseq==0.5.2 BioPython pandas Cython

git clone https://github.com/kathyxchen/selene.git
pip install selene-sdk
cd selene
git checkout custom_target_support
python setup.py build_ext --inplace
python setup.py install
cd ..


pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 torchtext==0.16.0 -f https://download.pytorch.org/whl/cu118/torch_stable.html


pip install natsort pytabix tqdm
echo "Setup complete. The environment is ready."
