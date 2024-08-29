# drGAT

[![arXiv](https://img.shields.io/badge/arXiv-2405.08979-b31b1b.svg)](https://arxiv.org/abs/2405.08979)

![](Figs/Fig1.png)

This is the official implementation for **drGAT: Attention-Guided Gene Assessment for Drug Response in Drug-Cell-Gene Heterogeneous Network**.  

This model is created to understand how genes influence Drug Response using Graph Attention Networks (GAT) on heterogeneous networks of drugs, cells, and genes. It predicts Drug Response based on the attention coefficients generated during this process. This has been implemented in Python.

## Quick start

This quick start allows you to run the drGAT training with the dataset on CPU and GPU. If you just would like to predict, run the preprocess and evaluation section. For demonstration purposese, the tutorial notebook runs a limited number of epochs, so the notebook run in a few minutes.

```shell
git clone git@github.com:inoue0426/drGAT.git
cd drGAT
docker pull inoue0426/drgat
docker run -it -p 9999:9999 inoue0426/drgat
```

Then access to http://localhost:9999/notebooks/Tutorial.ipynb 


## Requirement

```
numpy==1.23.5
pandas==2.0.3
matplotlib==3.7.1
optuna==3.2.0
torch==1.13.1+cu116
torch-cluster==1.6.1+pt113cu116
torch-geometric==2.3.1
torch-scatter==2.1.1+pt113cu116
torch-sparse==0.6.17+pt113cu116
torch-spline-conv==1.2.2+pt113cu116
```

## Environment

Our experiment was conducted on Ubuntu with an NVIDIA A100 Tensor Core GPU.  
If you want to re-train model, we reccomend to use GPU.
