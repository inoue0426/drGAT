import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.amp import GradScaler, autocast
from torch.nn import Dropout, Linear, Module
from torch.optim import lr_scheduler
from torch_geometric.nn import GCNConv, GraphNorm, MessagePassing
from tqdm import tqdm


class MPNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.lin = Linear(in_channels + 1, out_channels)  # +1 for edge_attr

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.lin(torch.cat([x_j, edge_attr], dim=-1))


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
        if self.gnn_layer == "GCN":
            self.gat1 = GCNConv(params["hidden1"], params["hidden2"])
            self.gat2 = GCNConv(params["hidden2"], params["hidden3"])
        elif self.gnn_layer == "MPNN":
            self.gat1 = MPNNConv(params["hidden1"], params["hidden2"])
            self.gat2 = MPNNConv(params["hidden2"], params["hidden3"])

        self.dropout1 = Dropout(params["dropout1"])
        self.dropout2 = Dropout(params["dropout2"])

        self.graph_norm1 = GraphNorm(params["hidden2"])
        self.graph_norm2 = GraphNorm(params["hidden3"])

        self.linear1 = Linear(params["hidden3"] + params["hidden3"], 1)

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

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1).to(torch.float32)

        if self.gnn_layer == "GCN":
            x = self.gat1(x=x, edge_index=edge_index).to(torch.float32)
            x = self.dropout1(self.activation(self.graph_norm1(x)))
            x = self.gat2(x=x, edge_index=edge_index).to(torch.float32)
            x = self.dropout2(self.activation(self.graph_norm2(x)))
        else:
            x = self.gat1(x=x, edge_index=edge_index, edge_attr=edge_attr).to(
                torch.float32
            )
            x = self.dropout1(self.activation(self.graph_norm1(x)))
            x = self.gat2(x=x, edge_index=edge_index, edge_attr=edge_attr).to(
                torch.float32
            )
            x = self.dropout2(self.activation(self.graph_norm2(x)))

        x = torch.concat(
            [
                x[idx_drug],
                x[idx_cell],
            ],
            1,
        )

        x = self.linear1(x)

        return x


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
            optimizer, step_size=params["step_size"], gamma=params["gamma_step"]
        ),
        "Plateau": lambda: lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=params["patience_plateau"],
            threshold=params["thresh_plateau"],
        ),
    }
    scheduler = schedulers.get(params["scheduler"], lambda: None)()

    # Initialize scaler
    scaler = GradScaler()

    return model, criterion, optimizer, scheduler, scaler


def train(
    sampler, params=None, is_sample=False, device=None, is_save=False, verbose=True
):
    # Set device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tensors = get_data_dict(sampler, device)

    # Initialize parameters
    params = initialize_params(
        params, tensors["drug"], tensors["cell"], tensors["gene"], is_sample
    )
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
        train_one_epoch(
            model,
            optimizer,
            criterion,
            scaler,
            tensors["drug"],
            tensors["cell"],
            tensors["gene"],
            tensors["edge_index"],
            tensors["edge_attr"],
            tensors["train_drug"],
            tensors["train_cell"],
            tensors["train_labels"],
            train_losses,
            train_accs,
            device,
        )

        val_acc, val_f1, val_auroc, val_aupr, val_labels, val_prob = validate_model(
            model,
            criterion,
            tensors["drug"],
            tensors["cell"],
            tensors["gene"],
            tensors["edge_index"],
            tensors["edge_attr"],
            tensors["val_drug"],
            tensors["val_cell"],
            tensors["val_labels"],
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
            best_val_labels, best_val_prob = val_labels, val_prob
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

        # Log output
        if verbose:
            print(
                "epoch:%4d" % (epoch + 1),
                "train_loss:%.6f" % train_losses[-1],
                "val_loss:%.6f" % val_losses[-1],
                "train_acc:%.4f" % train_accs[-1],
                "val_acc:%.4f" % val_accs[-1],
            )

    if best_epoch is not None:
        print(f"Best model found at epoch {best_epoch}")

    if is_save and best_model_state is not None:
        torch.save(best_model_state, "best_model_CTRP.pt")

    model.load_state_dict(best_model_state)

    return (
        model,
        best_val_labels,
        best_val_prob,
        best_metrics,
        early_stopping_epoch,
        val_labels,
        val_prob,
    )


def get_data_dict(sampler, device):
    # Move tensors to device
    return {
        "drug": sampler.S_d.to(device),
        "cell": sampler.S_c.to(device),
        "gene": sampler.S_g.to(device),
        "edge_index": sampler.edge_index.to(device),
        "edge_attr": sampler.edge_attr.to(device),
        "train_drug": torch.tensor(sampler.train_labels_df["Drug"].values).to(device),
        "train_cell": torch.tensor(sampler.train_labels_df["Cell"].values).to(device),
        "train_labels": torch.tensor(sampler.train_labels_df["Label"].values).to(
            device
        ),
        "val_drug": torch.tensor(sampler.test_labels_df["Drug"].values).to(device),
        "val_cell": torch.tensor(sampler.test_labels_df["Cell"].values).to(device),
        "val_labels": torch.tensor(sampler.test_labels_df["Label"].values).to(device),
    }



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

#     with autocast(device_type=device.type):
    outputs = model(drug, cell, gene, edge_index, edge_attr, train_drug, train_cell)
    loss = criterion(outputs.squeeze(), train_labels.float())

    train_losses.append(loss.item())

    predict = (torch.sigmoid(outputs) > 0.5).float().squeeze()
    train_acc = (predict == train_labels).sum().item() / len(predict)
    train_accs.append(train_acc)

    # train_one_epoch関数内
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # スケーリング解除後にクリッピング
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 勾配クリッピング
    scaler.step(optimizer)
    scaler.update()


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
#         with autocast(device_type=device.type):
        outputs = model(drug, cell, gene, edge_index, edge_attr, val_drug, val_cell)

        # NaNチェック
        if torch.isnan(outputs).any():
            print("NaN detected in model outputs!")
            print("Outputs:", outputs)

        loss = criterion(outputs.squeeze(), val_labels.float())
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
    return val_acc, val_f1, val_auroc, val_aupr, val_labels, probabilities


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
            outputs = model(
                drug, cell, gene, edge_index, edge_attr, test_drug, test_cell
            )

    probability = outputs.squeeze().float()
    predict = torch.round(outputs).squeeze().float()

    res = print_binary_classification_metrics(
        test_labels.cpu().detach().numpy(), predict.cpu().detach().numpy()
    )

    return probability, res
