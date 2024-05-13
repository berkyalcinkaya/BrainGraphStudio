# BrainGraphStudio
## An AutoML ToolKit for Classification of Static Functional Brain Graphs
### Developed for Atrium Health's [Laboratory for Complex Brain Networks](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://lcbn.wakehealth.edu/&ved=2ahUKEwiK8uybx4uGAxXoIDQIHa_LCJYQFnoECBUQAQ&usg=AOvVaw3vyg7-YjvvOsc1d6_8KYaD)
### Developer: Berk Yalcinkaya

[![PyPI version](https://badge.fury.io/py/BrainGraphStudio.svg)](https://badge.fury.io/py/BrainGraphStudio)
[![Downloads](https://pepy.tech/badge/BrainGraphStudio)](https://pepy.tech/project/BrainGraphStudio)
[![Downloads](https://pepy.tech/badge/BrainGraphStudio/month)](https://pepy.tech/project/BrainGraphStudio)
[![Python version](https://img.shields.io/pypi/pyversions/BrainGraphStudio)](https://pypistats.org/packages/BrainGraphStudio)
[![License: GPL v3](https://img.shields.io/github/license/berkyalcinkaya/BrainGraphStudio)](https://github.com/berkyalcinkaya/BrainGraphStudio/blob/main/LICENSE)
[![Contributors](https://img.shields.io/github/contributors-anon/berkyalcinkaya/BrainGraphStudio)](https://github.com/berkyalcinkaya/BrainGraphStudio/graphs/contributors)
[![repo size](https://img.shields.io/github/repo-size/berkyalcinkaya/BrainGraphStudio)](https://github.com/berkyalcinkaya/BrainGraphStudio/)
[![GitHub stars](https://img.shields.io/github/stars/berkyalcinkaya/BrainGraphStudio?style=social)](https://github.com/berkyalcinkaya/BrainGraphStudio/)
[![GitHub forks](https://img.shields.io/github/forks/berkyalcinkaya/BrainGraphStudio?style=social)](https://github.com/berkyalcinkaya/BrainGraphStudio/)

## Overview
BrainGraphStudio is a GUI-based tool for training, building, and optimizing BrainGNN[[1]](#1) or BrainGB[[2]](#2) graph neural networks.

## Install Instructions
`BrainGraphStudio` can be installed for CPU or GPU usage as follow. To download:

1. Install an [Anaconda](https://www.anaconda.com/products/distribution) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt/command prompt
3. If you have an older `bgs` environment you should remove it with `conda env remove -n bgs` before creating a new one. 
4. Create a new environment with `conda create --name bgs python=3.9.0`. 
5. Activate this new environment by running `conda activate bgs`
6. To download our package plus all dependencies, run `python -m pip install bgs[gpu]` on Windows and `python3 -m pip install bgs[gpu]` on Linux, Ubuntu, and Mac OS. Replace `gpu` with `cpu` if you intend to run BrainGraphStudio without GPU.

Next, run the following commands:
```pip install torch-cluster==1.5.9
pip install torch-scatter==2.0.8
pip install torch-sparse==0.6.12
pip install torch-spline-conv==1.2.1
pip install torch-geometric==2.0.4
```

## Running BrainGraphStudio
To run BrainGraphStudio, open the terminal, activate your bgs conda environment and run the following command
```
bgs
````

This should open the UI window and prompt you to load your data, configure the model architecture, and define hyperparameters


## References
<a id="1">[1]</a> 
Xiaoxiao Li, Yuan Zhou, Nicha Dvornek, Muhan Zhang, Siyuan Gao, Juntang Zhuang, Dustin Scheinost, Lawrence H. Staib, Pamela Ventola, James S. Duncan,
BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis,
Medical Image Analysis,
Volume 74,
2021,
102233,
ISSN 1361-8415,
https://doi.org/10.1016/j.media.2021.102233.

<a id="2">[2]</a> 
Cui, H., Dai, W., Zhu, Y., Kan, X., Chen Gu, A. A., Lukemire, J., Zhan, L., He, L., Guo, Y., & Yang, C. (2022). BrainGB: A Benchmark for Brain Network Analysis with Graph Neural Networks. IEEE Transactions on Medical Imaging (TMI).
