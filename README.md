# drGAT

[![arXiv](https://img.shields.io/badge/arXiv-2405.08979-b31b1b.svg)](https://arxiv.org/abs/2405.08979)

![](Figs/Fig1.png)

This is the official implementation for **drGAT: Attention-Guided Gene Assessment for Drug Response in Drug-Cell-Gene Heterogeneous Network**.  

This model is created to understand how genes influence Drug Response using Graph Attention Networks (GAT) on heterogeneous networks of drugs, cells, and genes. It predicts Drug Response based on the attention coefficients generated during this process. This has been implemented in Python.

## Quick start

This quick start allows you to run the drGAT training with the dataset on CPU and GPU. If you just would like to predict, run the preprocess and evaluation section. For demonstration purposes, the tutorial notebook runs a limited number of epochs, so the notebook run in a few minutes.

```shell
git clone git@github.com:inoue0426/drGAT.git
cd drGAT
docker pull inoue0426/drgat
docker run -it -p 9999:9999 inoue0426/drgat
```

Then access to http://localhost:9999/notebooks/Tutorial.ipynb 


## Training

For re-training the model, refer to model_training.ipynb. If you want to use your dataset, create_dataset.ipynb might be useful.


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
If you want to re-train model, we recommend using GPU.

---

## Installation using Docker

```shell
git clone git@github.com:inoue0426/drGAT.git
cd drGAT
docker build -t drgat:latest .
docker run -it -p 9999:9999 drgat:latest
```

Then access to http://localhost:9999/notebooks/Tutorial.ipynb 

## Installation using Conda

```shell
git clone git@github.com:inoue0426/drGAT.git
cd drGAT
conda env create -f environment.yml
conda activate drGAT
```
** NOTE: Please ensure the version matches exactly with your GPU/CPU specifications.


## Installation using requirement.txt

```shell
git clone git@github.com:inoue0426/drGAT.git
cd drGAT
conda create --name drGAT python=3.10 -y
conda activate drGAT
pip install -r requirement.txt
# Please make sure to change the version to match the version of your GPU/CPU machine exactly.
pip install --no-cache-dir  torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --no-cache-dir torch_geometric
pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1%2Bcu116.html
```
** NOTE: Please ensure the version matches exactly with your GPU/CPU specifications.
