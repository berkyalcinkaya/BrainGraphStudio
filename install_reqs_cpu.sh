#!/bin/bash

# Install PyTorch and related packages first
pip install torch==1.8.0
pip install torchvision==0.9.1
pip install torchaudio==0.8.1

# Install other dependencies
pip install numpy~=1.21.2
pip install pandas~=1.2.3
pip install scikit-learn~=1.0.2
pip install networkx~=2.6.2
pip install node2vec~=0.4.4
pip install scipy~=1.7.3
pip install nni~=2.4
pip install torch-cluster==1.5.9
pip install torch-scatter==2.0.8
pip install torch-sparse==0.6.12
pip install torch-spline-conv==1.2.1
pip install torch-geometric~=2.0.4