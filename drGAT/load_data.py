import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .utility import (get_morgan_fingerprint, min_max_scale, natural_sort_key,
                      normalize_similarity_matrix)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGS_DIR = PROJECT_ROOT / "Figs"


def load_data(data=None, is_zero_pad=True, verbose=False):
    dataset_paths = {
        "gdsc1": "gdsc1_data",
        "gdsc2": "gdsc2_data",
        "ctrp": "ctrp_data",
        "nci": "nci_data",
    }

    if data not in dataset_paths:
        raise NotImplementedError(f"Dataset {data} not implemented")

    print(f"load {data}")
    path = DATA_DIR / dataset_paths[data]
    return _load_dataset(path, is_zero_pad, verbose, data == "nci")


def _get_base_data(path):
    drugAct = pd.read_csv(path / "drugAct.csv", index_col=0)
    exprs = pd.concat(
        [
            pd.read_csv(f, index_col=0).T
            for f in sorted(
                path.glob("gene_exp_part*.csv.gz"),
                key=lambda x: natural_sort_key(str(x)),
            )
        ],
        axis=1,
    ).fillna(0)
    return drugAct, exprs


def _process_gene_expression(exprs, dti, verbose=False):
    selected_genes = sorted(
        set(exprs.std()[exprs.std() > np.percentile(exprs.std(), 90)].index)
        | set(dti["Gene"])
    )
    if verbose:
        print(
            f"Top 90% variable genes: \t{len(selected_genes) - len(set(dti['Gene']))}"
        )
        print(f"Total selected genes: \t{len(selected_genes)}")
    # else:
    #     print(f"Selected {len(selected_genes)} genes")
    return exprs[selected_genes]


def _get_drug_features(drugAct, smiles_data, smiles_key="SMILES", drug_key=None):
    smiles_dict = (
        dict(smiles_data[[drug_key, smiles_key]].values)
        if not isinstance(smiles_data, dict)
        else smiles_data
    )
    return pd.DataFrame(
        get_morgan_fingerprint([smiles_dict[i] for i in drugAct.index]),
        index=drugAct.index,
    )


def _get_normalized_gene_data(exprs):
    scaler = StandardScaler()
    return (
        pd.DataFrame(
            scaler.fit_transform(exprs), index=exprs.index, columns=exprs.columns
        ),
        pd.DataFrame(
            scaler.fit_transform(exprs.T), index=exprs.columns, columns=exprs.index
        ).T,
    )


def _print_dataset_stats(drugAct, cells, verbose):
    if verbose:
        print(f"unique drugs\t{len(drugAct.index)}")
        print(f"unique cells\t{len(cells)}")
        print(f"unique drug response\t{drugAct.notna().sum().sum()}")
        print(f"n sensitive\t{(drugAct == 1).sum().sum()}")
        print(f"n resistant\t{(drugAct == 0).sum().sum()}")
        drug_sensitive_ratio = (drugAct == 1).sum(axis=1) / drugAct.notna().sum(axis=1)
        cell_sensitive_ratio = (drugAct == 1).sum(axis=0) / drugAct.notna().sum(axis=0)
        print(f"AVG Drug binary ratio\t{drug_sensitive_ratio.mean():.3f}")
        print(f"AVG Cell binary ratio\t{cell_sensitive_ratio.mean():.3f}")
        print(f"Over 10 entries (drugs)\t{(drugAct.notna().sum(axis=1) > 9).sum()}")
        print(f"Over 10 entries (cells)\t{(drugAct.notna().sum(axis=0) > 9).sum()}")
    # else:
    #     print(f"Dataset size: {len(drugAct.index)} drugs x {len(cells)} cells")
    #     print(f"Total responses: {drugAct.notna().sum().sum()}")
    #     print(
    #         f"Responses distribution: {(drugAct == 1).sum().sum()} sensitive, {(drugAct == 0).sum().sum()} resistant"
    #     )


def _process_dti_data(drugAct, exprs, verbose, is_nci):
    dti = pd.read_csv(DATA_DIR / "full_table.csv")
    drug_col = "NSC" if is_nci else "Drug Name"
    if is_nci:
        dti = dti[
            dti[drug_col].isin(drugAct.index)
            & dti.Gene.isin(set(exprs.columns) & set(dti.Gene))
        ]
        dti.to_csv('nci_dti.csv')
    else:
        # Normalize drug names by removing special characters and converting to lowercase
        normalized_drugAct = {
            i.replace("-", "")
            .replace(".", "")
            .replace("/", "")
            .replace(" ", "")
            .lower()
            for i in drugAct.index
        }
        # Create mapping between normalized and original names
        drug_map = {
            i.replace("-", "")
            .replace(".", "")
            .replace("/", "")
            .replace(" ", "")
            .lower(): i
            for i in drugAct.index
        }
        # Filter DTI data based on normalized drug names
        dti = dti[
            dti[drug_col].apply(
                lambda x: x.replace("-", "")
                .replace(".", "")
                .replace("/", "")
                .replace(" ", "")
                .lower()
                in normalized_drugAct
            )
            & dti.Gene.isin(set(exprs.columns) & set(dti.Gene))
        ]
        # Update drug names to match drugAct index format
        dti[drug_col] = dti[drug_col].apply(
            lambda x: drug_map[
                x.replace("-", "")
                .replace(".", "")
                .replace("/", "")
                .replace(" ", "")
                .lower()
            ]
        )
    if verbose:
        print(f"dtis\t{len(dti)}")
        print(f"unique drugs\t{len(set(dti[drug_col]))}")
        print(f"unique genes\t{len(set(dti.Gene))}")
    return dti


def _create_drug_gene_adjacency(drugAct, A_cg, dti, is_zero_pad, is_nci):
    A_dg = pd.DataFrame(
        (
            np.zeros([len(drugAct.index), len(A_cg.columns)])
            if is_zero_pad
            else np.ones([len(drugAct.index), len(A_cg.columns)]) / 2
        ),
        index=drugAct.index,
        columns=A_cg.columns,
    )
    for _, i in dti.iterrows():
        drug_col = "NSC" if is_nci else "Drug Name"
        A_dg.loc[i[drug_col], i["Gene"]] = 1
    return A_dg


def _convert_to_tensors(*args):
    return tuple(torch.tensor(x.values).float() for x in args)


def _calculate_similarity_matrix(data, axis="cell"):
    return normalize_similarity_matrix(data.T if axis == "gene" else data)


def _load_dataset(path, is_zero_pad, verbose, is_nci):
    # Load base data
    drugAct, exprs = _get_base_data(path)
    smiles_data = pd.read_csv(
        (
            DATA_DIR / "nsc_cid_smiles_class_name.csv"
            if is_nci
            else path / "drug2smiles.csv"
        ),
        index_col=0,
    )

    # Filter and align data
    cells = sorted(
        set(drugAct.columns)
        & set(exprs.index)
        & (
            set(pd.read_csv(path / "mut.csv", index_col=0).T.index)
            if is_nci
            else set(exprs.index)
        )
    )
    drugs = sorted(set(drugAct.index) & set(smiles_data["NSC" if is_nci else "Drug"]))
    exprs, drugAct = exprs.loc[cells], drugAct.loc[drugs, cells]
    smiles_data = smiles_data[smiles_data["NSC" if is_nci else "Drug"].isin(drugs)]

    # Assert data consistency
    assert set(drugAct.columns) == set(
        exprs.index
    ), "DrugAct columns and Gene Exp index mismatch"
    assert set(drugAct.index) == set(
        smiles_data["NSC" if is_nci else "Drug"]
    ), "DrugAct index and SMILES data mismatch"

    _print_dataset_stats(drugAct, cells, verbose)

    # Process features and similarities
    drug_feature = _get_drug_features(
        drugAct, smiles_data, "SMILES", "NSC" if is_nci else "Drug"
    )
    drug_sim = normalize_similarity_matrix(drug_feature)
    assert drug_sim.index.equals(
        drug_sim.columns
    ), "Drug similarity matrix should be symmetric"
    assert drug_sim.index.equals(
        drugAct.index
    ), "Drug similarity matrix index mismatch with drugAct"

    dti = _process_dti_data(drugAct, exprs, verbose, is_nci)
    assert set(dti["NSC" if is_nci else "Drug Name"]).issubset(
        set(drugAct.index)
    ), "DTI contains drugs not in drugAct"
    assert set(dti["Gene"]).issubset(
        set(exprs.columns)
    ), "DTI contains genes not in gene expression data"

    exprs = _process_gene_expression(exprs, dti, verbose)

    gene_norm_cell, gene_norm_gene = _get_normalized_gene_data(exprs)
    gene_sim = _calculate_similarity_matrix(gene_norm_cell, axis="gene")
    cell_sim = _calculate_similarity_matrix(gene_norm_gene, axis="cell")

    # Assert similarity matrices consistency
    assert gene_sim.index.equals(
        gene_sim.columns
    ), "Gene similarity matrix should be symmetric"
    assert cell_sim.index.equals(
        cell_sim.columns
    ), "Cell similarity matrix should be symmetric"
    assert gene_sim.index.equals(
        exprs.columns
    ), "Gene similarity matrix index mismatch with gene expression"
    assert cell_sim.index.equals(
        exprs.index
    ), "Cell similarity matrix index mismatch with gene expression"

    # Create adjacency matrices
    A_cg = gene_norm_cell * (gene_norm_cell > 0).astype(int)
    A_dg = _create_drug_gene_adjacency(drugAct, A_cg, dti, is_zero_pad, is_nci)

    # Assert adjacency matrices consistency
    assert A_cg.index.equals(exprs.index), "A_cg index mismatch with gene expression"
    assert A_cg.columns.equals(
        exprs.columns
    ), "A_cg columns mismatch with gene expression"
    assert A_dg.index.equals(drugAct.index), "A_dg index mismatch with drugAct"
    tensors = _convert_to_tensors(
        drug_sim, cell_sim, gene_sim, drug_feature, gene_norm_gene, gene_norm_cell
    )

    print("Done!")
    return (
        drugAct.T,
        (drugAct.isna()).astype(int).T,
        *tensors,
        A_cg,
        A_dg,
    )


# How to use
# res, null_mask, S_d, S_c, S_g, drug_feature, gene_norm_gene, gene_norm_cell, A_cg, A_dg = load_data(data="nci", is_zero_pad=True, verbose=True)
