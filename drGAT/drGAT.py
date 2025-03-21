import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.amp import GradScaler, autocast
from torch.nn import Dropout, Linear, Module
from torch.optim import lr_scheduler
from torch_geometric.nn import GATConv, GATv2Conv, GraphNorm, TransformerConv
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


class drGAT(Module):
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
        super(drGAT, self).__init__()
        self.linear_drug = Linear(params["n_drug"], params["hidden1"])
        self.linear_cell = Linear(params["n_cell"], params["hidden1"])
        self.linear_gene = Linear(params["n_gene"], params["hidden1"])

        self.gnn_layer = params["gnn_layer"]
        if self.gnn_layer == "GAT":
            self.gat1 = GATConv(
                params["hidden1"], params["hidden2"], heads=params["heads"], edge_dim=1
            )
            self.gat2 = GATConv(
                params["hidden2"] * params["heads"],
                params["hidden3"],
                heads=params["heads"],
                edge_dim=1,
            )
        elif self.gnn_layer == "GATv2":
            self.gat1 = GATv2Conv(
                params["hidden1"], params["hidden2"], heads=params["heads"], edge_dim=1
            )
            self.gat2 = GATv2Conv(
                params["hidden2"] * params["heads"],
                params["hidden3"],
                heads=params["heads"],
                edge_dim=1,
            )
        elif self.gnn_layer == "Transformer":
            self.gat1 = TransformerConv(
                params["hidden1"], params["hidden2"], heads=params["heads"], edge_dim=1
            )
            self.gat2 = TransformerConv(
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
            params["hidden3"] * params["heads"] + params["hidden3"] * params["heads"],
            1,
        )

        self.activation = self._get_activation(params.get("activation", "relu"))

    def _get_activation(self, name):
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),  # PyTorch 1.7+で正式実装
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, drug, cell, gene, edge_index, edge_attr, idx_drug, idx_cell):

        x = torch.concat(
            [self.linear_drug(drug), self.linear_cell(cell), self.linear_gene(gene)]
        )

        if self.gnn_layer == "Transformer":
            edge_attr = edge_attr.unsqueeze(-1)

        x, attention = self.gat1(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True,
        )
        all_attention = get_attention_mat(attention)
        del attention

        x = x.to(torch.float32)
        x = self.dropout1(self.activation(self.graph_norm1(x)))

        x, attention = self.gat2(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True,
        )
        all_attention += get_attention_mat(attention)
        del attention

        x = x.to(torch.float32)
        x = self.dropout2(self.activation(self.graph_norm2(x)))

        x = torch.concat(
            [
                x[idx_drug],
                x[idx_cell],
            ],
            1,
        )

        x = self.linear1(x)

        return x, all_attention


def get_model(params, device):
    """
    A function to get a model.
    params: contains params for the model and optimizer
    device: device to use
    """
    # Initialize model and criterion
    model = drGAT(params).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Select optimizer class dynamically
    optimizer_class = getattr(torch.optim, params["optimizer"])

    # Set optimizer parameters
    optimizer_kwargs = {
        "params": model.parameters(),
        "lr": params["lr"],
        "weight_decay": params.get("weight_decay", 0.0),
    }

    # Additional parameters for Adam-based optimizers
    if params["optimizer"].lower() in ["adam", "adamw"]:
        optimizer_kwargs.update({"amsgrad": params.get("amsgrad", False)})

    # Additional parameters for SGD optimizer
    if params["optimizer"].lower() == "sgd":
        optimizer_kwargs.update(
            {
                "momentum": params.get("momentum", 0.9),
                "nesterov": params.get("nesterov", True),
            }
        )

    # Initialize optimizer
    optimizer = optimizer_class(**optimizer_kwargs)

    # Scheduler selection
    schedulers = {
        "Cosine": lambda: lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=params["T_max"]
        ),
        "Step": lambda: lr_scheduler.StepLR(
            optimizer, step_size=params["step_size"], gamma=params["scheduler_gamma"]
        ),
        "Plateau": lambda: lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=params["patience"],
            threshold=params["threshold"],
        ),
    }
    scheduler = schedulers.get(params["scheduler"], lambda: None)()

    # Initialize scaler
    scaler = GradScaler()

    return model, criterion, optimizer, scheduler, scaler


def train(data, params=None, is_sample=False, device=None, is_save=False, verbose=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    data = [x.to(device) if torch.is_tensor(x) else x for x in data]
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
    ) = data

    params = initialize_params(params, drug, cell, gene, is_sample)

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    model, criterion, optimizer, scheduler, scaler = get_model(params, device)

    best_model_state = None
    early_stopping_counter = 0
    max_early_stopping = 10
    early_stopping_epoch = None
    epoch_range = range(params["epochs"])
    if not verbose:
        epoch_range = tqdm(epoch_range)

    for epoch in epoch_range:
        train_attention = train_one_epoch(
            model,
            optimizer,
            criterion,
            scaler,
            drug,
            cell,
            gene,
            edge_index,
            edge_attr,
            train_drug,
            train_cell,
            train_labels,
            train_losses,
            train_accs,
            device,
        )

        val_acc, val_f1, val_auroc, val_aupr, val_attention = validate_model(
            model,
            criterion,
            drug,
            cell,
            gene,
            edge_index,
            edge_attr,
            val_drug,
            val_cell,
            val_labels,
            val_losses,
            val_accs,
            device,
        )

        # スケジューラの更新
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses[-1])
            else:
                scheduler.step()

        best_metrics = [
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # [best_val_acc, best_val_aupr, best_val_auroc, best_val_f1]
        best_epoch = None

        if val_acc > best_metrics[0]:
            best_metrics = [val_acc, val_aupr, val_auroc, val_f1]
            best_model_state = model.state_dict()
            best_train_attention = train_attention
            best_val_attention = val_attention
            best_epoch = epoch + 1
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= max_early_stopping:
            print(
                f"Early stopping at epoch {epoch + 1} because validation accuracy did not improve for {max_early_stopping} epochs."
            )
            early_stopping_epoch = epoch + 1
            break

        if verbose:
            print(
                f"Epoch {epoch + 1}: Train Loss = {round(train_losses[-1], 4)}, Val Loss = {round(val_losses[-1], 4)}, Train Acc = {round(train_accs[-1], 4)}, \nVal Acc = {round(val_accs[-1], 4)}, Val F1 = {round(val_f1, 4)}, Val AUROC = {round(val_auroc, 4)}, Val AUPR = {round(val_aupr, 4)}"
            )

    if best_epoch is not None:
        print(f"Best model found at epoch {best_epoch}")

    if is_save and best_model_state is not None:
        torch.save(best_model_state, "best_model_CTRP.pt")

    model.load_state_dict(best_model_state)

    return (
        model,
        best_train_attention,
        best_val_attention,
        best_metrics,
        early_stopping_epoch,
    )


def initialize_params(params, drug, cell, gene, is_sample):
    params = params or {
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
    return params


def train_one_epoch(
    model,
    optimizer,
    criterion,
    scaler,
    drug,
    cell,
    gene,
    edge_index,
    edge_attr,
    train_drug,
    train_cell,
    train_labels,
    train_losses,
    train_accs,
    device,
):
    model.train()
    optimizer.zero_grad()

    with autocast(device_type=device.type):
        outputs, attention = model(
            drug, cell, gene, edge_index, edge_attr, train_drug, train_cell
        )
        loss = criterion(outputs.squeeze(), train_labels)

    train_losses.append(loss.item())

    predict = (torch.sigmoid(outputs) > 0.5).float().squeeze()
    train_acc = (predict == train_labels).sum().item() / len(predict)
    train_accs.append(train_acc)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return attention


def validate_model(
    model,
    criterion,
    drug,
    cell,
    gene,
    edge_index,
    edge_attr,
    val_drug,
    val_cell,
    val_labels,
    val_losses,
    val_accs,
    device,
):
    model.eval()
    with torch.no_grad():
        with autocast(device_type=device.type):
            outputs, attention = model(
                drug, cell, gene, edge_index, edge_attr, val_drug, val_cell
            )
            loss = criterion(outputs.squeeze(), val_labels)
        val_losses.append(loss.item())

        outputs = outputs.squeeze().float().cpu()  # ここで次元を調整
        probabilities = torch.sigmoid(outputs).numpy()
        predict = (probabilities > 0.5).astype(int)
        val_labels = val_labels.cpu().numpy()
        val_acc = (predict == val_labels).sum().item() / len(predict)
        val_accs.append(val_acc)
        val_f1 = f1_score(val_labels, predict)
        val_auroc = roc_auc_score(val_labels, probabilities)
        val_aupr = average_precision_score(val_labels, probabilities)
    return val_acc, val_f1, val_auroc, val_aupr, attention


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
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    drug, cell, gene, edge_index, edge_attr, test_drug, test_cell, test_labels = [
        x.to(device) if torch.is_tensor(x) else x for x in data
    ]

    model.eval()
    with torch.no_grad():
        with autocast(device_type=device.type):
            outputs, test_attention = model(
                drug, cell, gene, edge_index, edge_attr, test_drug, test_cell
            )

    probability = outputs.squeeze().float()
    predict = torch.round(outputs).squeeze().float()

    res = print_binary_classification_metrics(
        test_labels.cpu().detach().numpy(), predict.cpu().detach().numpy()
    )

    return probability, res, test_attention
