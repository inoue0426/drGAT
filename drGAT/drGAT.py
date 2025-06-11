import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from packaging import version
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.nn import Dropout, Linear, Module
from torch.optim import lr_scheduler
from torch_geometric.nn import GATConv, GATv2Conv, GraphNorm, TransformerConv
from tqdm import tqdm

# Compatibility handling for AMP (Automatic Mixed Precision)
if version.parse(torch.__version__) >= version.parse("1.10"):
    from torch.amp import autocast  # autocast is imported here for common use

    if torch.cuda.is_available():
        from torch.cuda.amp import GradScaler

        autocast_device = "cuda"
        use_autocast = True
    else:
        GradScaler = lambda: None  # dummy scaler
        autocast_device = "cpu"
        use_autocast = True
else:
    # fallback: for environments where autocast is not available
    class DummyAutocast:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    autocast = DummyAutocast
    GradScaler = lambda: None
    use_autocast = False

    autocast = DummyAutocast
    GradScaler = lambda: None  # dummy scaler
    use_autocast = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def get_attention_mat(attention):

    tmp = attention[1].cpu().detach().numpy()
    edge = attention[0].cpu().detach().numpy()

    idx = edge.max() + 1
    graph = np.zeros((idx, idx))
    graph[edge[0], edge[1]] = tmp.mean(axis=1)

    return graph


class drGAT(Module):
    def __init__(self, params):
        super(drGAT, self).__init__()

        # Initialize hidden layer dimensions and number of attention heads
        hidden1: int = int(params["hidden1"])
        hidden2: int = int(params["hidden2"])
        hidden3: int = int(params["hidden3"])
        heads: int = int(params["heads"])
        self.n_layers: int = int(params.get("n_layers", 2))  # Default is 2 layers

        # Linear layers for initial feature transformation
        self.linear_drug = Linear(int(params["n_drug"]), hidden1)
        self.linear_cell = Linear(int(params["n_cell"]), hidden1)
        self.linear_gene = Linear(int(params["n_gene"]), hidden1)

        # Store GNN layer type
        self.gnn_layer = params["gnn_layer"]

        # Initialize lists for GNN layers, normalization, and dropout
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.attn_dropout = params.get("attention_dropout", 0.0)
        self.final_mlp_layers = params.get("final_mlp_layers", 1)

        # Configure dimensions for each layer
        in_channels = [hidden1] + [hidden2 * heads] * (self.n_layers - 1)
        out_channels = [hidden2] * (self.n_layers - 1) + [hidden3]

        # Create GNN layers based on specified type
        for i in range(self.n_layers):
            if self.gnn_layer == "GAT":
                self.gat_layers.append(
                    GATConv(
                        in_channels[i],
                        out_channels[i],
                        heads=heads,
                        edge_dim=1,
                        dropout=self.attn_dropout,
                    )
                )
            elif self.gnn_layer == "GATv2":
                self.gat_layers.append(
                    GATv2Conv(
                        in_channels[i],
                        out_channels[i],
                        heads=heads,
                        edge_dim=1,
                        dropout=self.attn_dropout,
                    )
                )
            elif self.gnn_layer == "Transformer":
                self.gat_layers.append(
                    TransformerConv(
                        in_channels[i],
                        out_channels[i],
                        heads=heads,
                        edge_dim=1,
                        dropout=self.attn_dropout,
                    )
                )

            # Add normalization and dropout layers
            if params["norm_type"] == "GraphNorm":
                self.norm_layers.append(GraphNorm(out_channels[i] * heads))
            elif params["norm_type"] == "BatchNorm":
                self.norm_layers.append(nn.BatchNorm1d(out_channels[i] * heads))
            elif params["norm_type"] == "LayerNorm":
                self.norm_layers.append(nn.LayerNorm(out_channels[i] * heads))

            self.dropouts.append(
                Dropout(params["dropout1"] if i == 0 else params["dropout2"])
            )

        # Set activation function
        self.activation = self._get_activation(params.get("activation", "relu"))

        # Final linear layer for prediction
        mlp_layers = []
        in_dim = 2 * hidden3 * heads
        for _ in range(self.final_mlp_layers - 1):
            mlp_layers.append(Linear(in_dim, in_dim))
            mlp_layers.append(self.activation)
            mlp_layers.append(Dropout(params["dropout3"]))
        mlp_layers.append(Linear(in_dim, 1))
        self.linear1 = nn.Sequential(*mlp_layers)

    def _get_activation(self, name):
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, drug, cell, gene, edge_index, edge_attr, idx_drug, idx_cell):
        # Concatenate transformed features
        x = torch.concat(
            [self.linear_drug(drug), self.linear_cell(cell), self.linear_gene(gene)]
        )

        # Adjust edge attributes for Transformer layer
        if self.gnn_layer == "Transformer":
            edge_attr = edge_attr.unsqueeze(-1)

        edge_attr = edge_attr.to(torch.float32)
        all_attention = 0

        # Apply GNN layers sequentially
        for i in range(self.n_layers):
            x, attention = self.gat_layers[i](
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                return_attention_weights=True,
            )
            
            tmp = get_attention_mat(attention)
            all_attention += tmp
            del attention

            # Apply normalization, activation, and dropout
            x = x.to(torch.float32)
            x = self.dropouts[i](self.activation(self.norm_layers[i](x)))

        # Extract and concatenate relevant node features
        x = torch.concat(
            [
                x[idx_drug],
                x[idx_cell],
            ],
            1,
        )

        # Final prediction
        x = self.linear1(x)

        return x, all_attention


def get_model(params, device):
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
    scaler = GradScaler() if use_autocast else None

    return model, criterion, optimizer, scheduler, scaler

def train(
    sampler,
    params=None,
    is_sample=False,
    device=None,
    is_save=False,
    verbose=True,
):
    # Set device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tensors = get_data_dict(sampler, device)

    # Initialize parameters
    params = initialize_params(
        params, tensors["drug"], tensors["cell"], tensors["gene"], is_sample
    )

    # Set up model and optimizer
    model, criterion, optimizer, scheduler, scaler = get_model(params, device)

    # Lists for recording losses
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_model_state = None

    # Set epoch range
    epoch_range = range(params["epochs"])
    # if verbose:
    #     epoch_range = tqdm(epoch_range)

    # Update best model
    best_metrics = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]  # [best_val_acc, best_val_aupr, best_val_auroc, best_val_f1]
    best_epoch = None
        
    for epoch in epoch_range:
        # Train one epoch
        train_attention = train_one_epoch(
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
            sampler, 
            params
        )

        # Validation
        val_acc, val_f1, val_auroc, val_aupr, val_attention, val_labels, val_prob = (
            validate_model(
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
        )

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses[-1])
            else:
                scheduler.step()


        
        if val_acc > best_metrics[0]:
            best_metrics = [val_acc, val_aupr, val_auroc, val_f1]
            best_model_state = model.state_dict()
            best_val_labels, best_val_prob = val_labels, val_prob
            best_epoch = epoch + 1

            best_train_attention = train_attention
            best_val_attention = val_attention

        # Log output
        if verbose and (epoch + 1) % 10 == 0:
            print(
                "epoch:%4d  train_loss:%.4f  val_loss:%.4f  train_acc:%.4f  val_acc:%.4f" % (
                    epoch + 1,
                    train_losses[-1],
                    val_losses[-1],
                    train_accs[-1],
                    val_accs[-1],
                )
            )
            
    # Save best model
    if best_epoch is not None:
        print(f"Best model found at epoch {best_epoch}")

    if is_save and best_model_state is not None:
        torch.save(best_model_state, "best_model.pt")

    model.load_state_dict(best_model_state)

    return (
        model,
        best_train_attention,
        best_val_attention,
        best_val_labels,
        best_val_prob,
        best_metrics,
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
        "heads": 2,
        "gnn_layer": "GATv2",
        "activation": "relu",
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "scheduler": None,
        "norm_type": "GraphNorm",
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
    sampler,
    params,
):
    model.train()
    optimizer.zero_grad()

    with autocast(autocast_device):
        outputs, attention = model(
            drug, cell, gene, edge_index, edge_attr, train_drug, train_cell
        )
        # 通常の分類タスクの損失
        loss = criterion(outputs.squeeze(), train_labels.float())

    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    with torch.no_grad():
        predict = (torch.sigmoid(outputs) > 0.5).float().squeeze()
        train_acc = (predict == train_labels).sum().item() / len(predict)
        train_losses.append(loss.item())
        train_accs.append(train_acc)

    del outputs, loss, predict
    torch.cuda.empty_cache()

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
        with autocast(autocast_device):
            outputs, attention = model(
                drug, cell, gene, edge_index, edge_attr, val_drug, val_cell
            )
            loss = criterion(outputs.squeeze(), val_labels.float())
        val_losses.append(loss.item())

        outputs = outputs.squeeze().float().cpu()
        probabilities = torch.sigmoid(outputs).numpy()
        predict = (probabilities > 0.5).astype(int)
        val_labels = val_labels.cpu().numpy()
        val_acc = (predict == val_labels).sum().item() / len(predict)
        val_accs.append(val_acc)
        val_f1 = f1_score(val_labels, predict)
        val_auroc = roc_auc_score(val_labels, probabilities)
        val_aupr = average_precision_score(val_labels, probabilities)
    return val_acc, val_f1, val_auroc, val_aupr, attention, val_labels, probabilities