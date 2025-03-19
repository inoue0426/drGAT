"""
This is the official implementation of "drGAT: Attention-Guided Gene Assessment
for Drug Response in Drug-Cell-Gene Heterogeneous Network."

Written by inoue0426
If you have any quesionts, feel free to make an issue to https://github.com/inoue0426/drGAT
"""

import subprocess

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.nn import Dropout, Linear, Module
from torch.nn.functional import relu
from torch_geometric.nn import GATv2Conv, GraphNorm
from tqdm import tqdm


def get_attention_mat(attention):
    """A function to make attention coefficient tensor to matrix.
    attention: attention tensor from Graph Attention layer.
    """

    edge_index, attention_weights = [i.detach().cpu() for i in attention]

    num_nodes = torch.max(edge_index) + 1
    attention_matrix = torch.zeros(
        (num_nodes, num_nodes), dtype=attention_weights.dtype
    )
    attention_matrix[edge_index[0], edge_index[1]] = attention_weights.squeeze().mean()

    return attention_matrix


class GAT(Module):
    """A class to generate a drGAT model.
    params: contains params for the model
        - dropout1: dropout rate for dropout 1
        - dropout2: dropout rate for dropout 2
        - hidden1: shape for the hidden1
        - hidden2: shape for the hidden2
        - hidden3: shape for the hidden3
        - heads: The number of heads for graph attention
    """

    def __init__(self, params):
        super(GAT, self).__init__()
        self.linear_drug = Linear(params["n_drug"], params["hidden1"])
        self.linear_cell = Linear(params["n_cell"], params["hidden1"])
        self.linear_gene = Linear(params["n_gene"], params["hidden1"])

        self.gat1 = GATv2Conv(
            params["hidden1"], params["hidden2"], heads=params["heads"], edge_dim=1
        )
        self.gat2 = GATv2Conv(
            params["hidden2"] * params["heads"],
            params["hidden3"],
            heads=params["heads"],
            edge_dim=1,
        )

        self.dropout1 = Dropout(params["dropout1"])
        self.dropout2 = Dropout(params["dropout2"])

        self.graph_norm1 = GraphNorm(params["hidden2"] * params["heads"])
        self.graph_norm2 = GraphNorm(params["hidden3"] * params["heads"])

        self.linear1 = Linear(
            params["hidden3"] * params["heads"] + params["hidden3"] * params["heads"], 1
        )

    def forward(self, drug, cell, gene, edge_index, edge_attr, idx_drug, idx_cell):

        x = torch.concat(
            [self.linear_drug(drug), self.linear_cell(cell), self.linear_gene(gene)]
        )

        x, attention = self.gat1(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True,
        )
        all_attention = get_attention_mat(attention)
        del attention

        x = self.dropout1(relu(self.graph_norm1(x)))

        x, attention = self.gat2(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True,
        )
        all_attention += get_attention_mat(attention)
        del attention

        x = self.dropout2(relu(self.graph_norm2(x)))

        x = torch.concat(
            [
                x[idx_drug],
                x[idx_cell],
            ],
            1,
        )

        x = self.linear1(x)

        return torch.sigmoid(x), all_attention


def get_model(params, device):
    """
    A function to get a model.
    params: contains params for the model
    device: device to use
    """

    model = GAT(params).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = getattr(torch.optim, "Adam")(
        model.parameters(),
        lr=params["lr"],
    )
    return model, criterion, optimizer


def train(data, params=None, is_sample=False, device=None, is_save=False):
    """
    Trains a model using the provided data and parameters.

    Parameters:
    - data: A tuple containing the following elements:
        - drug: Feature matrix for drug nodes.
        - cell: Feature matrix for cell nodes.
        - gene: Feature matrix for gene nodes.
        - edge_index: Adjacency matrix for the graph.
        - edge_attr: Edge attributes for the graph.
        - train_drug: Indices of drug nodes used for training.
        - train_cell: Indices of cell nodes used for training.
        - train_labels: Labels corresponding to the training data.
        - val_drug: Indices of drug nodes used for validation.
        - val_cell: Indices of cell nodes used for validation.
        - val_labels: Labels corresponding to the validation data.
    - params: Dictionary containing parameters for the model.
    - is_sample: Boolean indicating whether to use a reduced number of epochs for training.
    - device: The device to use for computation (e.g., 'cpu' or 'cuda').
    - is_save: Boolean indicating whether to save the best model during training.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using: ", device)

    (
        drug,
        cell,
        gene,
        edge_index,
        edge_attr,
        train_drug,
        train_cell,
        val_drug,
        val_cell,
        train_labels,
        val_labels,
    ) = [x.to(device) if torch.is_tensor(x) else x for x in data]

    if not params:
        params = {
            "dropout1": 0.1,
            "dropout2": 0.1,
            "n_drug": drug.shape[0],
            "n_cell": cell.shape[0],
            "n_gene": gene.shape[0],
            "hidden1": 256,
            "hidden2": 32,
            "hidden3": 128,
            "epochs": 1500,
            "lr": 0.001,
            "heads": 5,
        }

    if is_sample:
        params["epochs"] = 5

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    model, criterion, optimizer = get_model(params, device)

    best_val_acc = 0.0
    early_stopping_counter = 0
    max_early_stopping = 10
    tmp = -1

    for epoch in tqdm(range(params["epochs"])):
        model.train()
        optimizer.zero_grad()

        outputs, attention = model(
            drug, cell, gene, edge_index, edge_attr, train_drug, train_cell
        )

        loss = criterion(outputs.squeeze(), train_labels)
        train_losses.append(loss.item())

        predict = torch.round(outputs).squeeze()
        train_acc = (predict == train_labels).sum().item() / len(predict)
        train_accs.append(train_acc)

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            outputs, _ = model(
                drug, cell, gene, edge_index, edge_attr, val_drug, val_cell
            )
            loss = criterion(outputs.squeeze(), val_labels)
            val_losses.append(loss.item())
            predict = torch.round(outputs).squeeze()
            val_acc = (predict == val_labels).sum().item() / len(predict)
            val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if is_save:
                    torch.save(model, "model_{}.pt".format(epoch))

                if tmp >= 0:
                    subprocess.run(["rm", "-rf", f"model_{tmp}.pt"], check=True)
                tmp = epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= max_early_stopping:
                print(
                    f"Early stopping at epoch {epoch + 1} because validation accuracy did not improve for {max_early_stopping} epochs."
                )
                break

        if (epoch + 1) % 10 == 0:
            print("Epoch: ", epoch + 1)
            print("Train Loss: ", train_losses[-1])
            print("Val Loss: ", val_losses[-1])
            print("Train Accuracy: ", train_accs[-1])
            print("Val Accuracy: ", val_accs[-1], "\n")

    return model, attention


def print_binary_classification_metrics(y_true, y_pred):
    """
    A function to print binary classification metrics.
    y_true: true labels
    y_pred: predicted labels
    """
    if len(y_true) == 1:
        return None

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics_data = {
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "True Positive": [tp],
        "True Negative": [tn],
        "False Positive": [fp],
        "False Negative": [fn],
    }

    metrics_df = pd.DataFrame(metrics_data)

    return metrics_df


def eval(model, data, device=None):
    """
    A function to evaluate a model.

    Parameters:
    model: torch.nn.Module
        The model to evaluate.
    data: tuple
        A tuple containing the data for evaluation:
        - drug: torch.Tensor
            Feature matrix for drug nodes.
        - cell: torch.Tensor
            Feature matrix for cell nodes.
        - gene: torch.Tensor
            Feature matrix for gene nodes.
        - edge_index: torch.Tensor
            Edge indices for the graph.
        - edge_attr: torch.Tensor
            Edge attributes for the graph.
        - test_drug: torch.Tensor
            Indices of drug nodes for testing.
        - test_cell: torch.Tensor
            Indices of cell nodes for testing.
        - test_labels: torch.Tensor
            Labels for testing.
    device: torch.device, optional
        The device to use for computation. If not provided, it will default to CUDA if available, otherwise CPU.
    """
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    drug, cell, gene, edge_index, edge_attr, test_drug, test_cell, test_labels = [
        x.to(device) if torch.is_tensor(x) else x for x in data
    ]

    model.eval()
    with torch.no_grad():
        outputs, _ = model(
            drug, cell, gene, edge_index, edge_attr, test_drug, test_cell
        )

    probability = outputs.squeeze()
    predict = torch.round(outputs).squeeze()

    res = print_binary_classification_metrics(
        test_labels.cpu().detach().numpy(), predict.cpu().detach().numpy()
    )

    return probability, res
