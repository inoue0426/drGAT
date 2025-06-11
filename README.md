# drGAT

[![arXiv](https://img.shields.io/badge/arXiv-2405.08979-b31b1b.svg)](https://arxiv.org/abs/2405.08979)

![](Figs/Fig1.png)

This is the official implementation for **drGAT: Attention-Guided Gene Assessment for Drug Response in Drug-Cell-Gene Heterogeneous Network**.  

This model is created to understand how genes influence Drug Response using Graph Attention Networks (GAT) on heterogeneous networks of drugs, cells, and genes. It predicts Drug Response based on the attention coefficients generated during this process. This has been implemented in Python.

## Quick start

This quick start guide demonstrates how to run drGAT predictions on both CPU and GPU, completing the process within seconds.

```shell
git clone git@github.com:inoue0426/drGAT.git
cd drGAT
docker build -t drgat:latest .
docker run -it -p 9999:9999 inoue0426/drgat
```

Then access http://localhost:9999/notebooks/Tutorial.ipynb and run all cells.

\* This might require you to increase the memory usage on docker.
If so, please follow this:
1. Open Docker Desktop Dashboard
2. Click on the Settings icon
3. Navigate to Resources > Advanced
4. Adjust the Memory slider to increase the limit
5. Click "Apply & Restart" to save changes

## Input

The model takes the following data structure (please refer to the [notebook](https://github.com/inoue0426/drGAT/blob/main/create_dateset.ipynb) for detailed information):

```python
data = [
    drug,          # Drug similarity matrix
    cell,          # Cell line similarity matrix
    gene,          # Gene similarity matrix
    edge_index,    # Graph edge indices
    train_drug,    # Training set drug indices
    train_cell,    # Training set cell line indices
    val_drug,      # Validation set drug indices
    val_cell,      # Validation set cell line indices
    train_labels,  # Training set binary labels
    val_labels     # Validation set binary labels
]
```

## Output

### For multiple drugs and cell lines

```python
predict, res = drGAT.eval(model, test) # Probability of sensitivity and Metrics.
res # Metrics

| Accuracy | Precision | Recall | F1 Score | True Positive | True Negative | False Positive | False Negative |
|-----------|-----------|---------|-----------|----------------|---------------|----------------|-----------------|
| 0.771375 | 0.740881 | 0.783245 | 0.761474 | 1178 | 1312 | 412 | 326 |

predict # Probability

tensor([0.7653, 0.3292, 0.3037,  ..., 0.9121, 0.4277, 0.2037])
```

### For single drug and cell line

```python
predict, _ = drGAT.eval(model, test)
predict

# Probability of sensitivity.
tensor(0.7653)
```


## Training

Refer to model_training.ipynb to retrain the model. If you want to use your dataset, create_dataset.ipynb might be useful.


## Requirement

```
    "torch==2.0.1",
    "torch-geometric==2.3.1",
    "numpy==1.26.4",
    "matplotlib",
    "pandas==2.2.2",
    "jupyter",
    "ipykernel"
```

** NOTE: Please ensure the version matches exactly with your GPU/CPU specifications.

## Environment

Our experiment was conducted on Ubuntu with an NVIDIA A100 Tensor Core GPU.  
If you want to re-train the model, we recommend using GPUs.

---

## Installation

```shell
uv venv --python 3.10
uv pip install -r pyproject.toml
jupyter notebook --port 9999
```

Then access to http://localhost:9999/notebooks/Tutorial.ipynb 

** NOTE: Please ensure the version matches exactly with your GPU/CPU specifications.

## Data

Data for this project came from [CellMinerCDB](https://pubmed.ncbi.nlm.nih.gov/30553813/) and is in the [data direcotry](https://github.com/inoue0426/drGAT/tree/main/data) as well as the preprocessing code [here](https://github.com/inoue0426/drGAT/tree/main/preprocess).

## Citation 

```
@article{inoue2024drgat,
  title={drGAT: Attention-Guided Gene Assessment of Drug Response Utilizing a Drug-Cell-Gene Heterogeneous Network},
  author={Inoue, Yoshitaka and Lee, Hunmin and Fu, Tianfan and Luna, Augustin},
  journal={ArXiv},
  year={2024},
  publisher={arXiv}
}
```
