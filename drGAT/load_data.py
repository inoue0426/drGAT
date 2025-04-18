import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .utility import (get_morgan_fingerprint, min_max_scale, natural_sort_key,
                      normalize_similarity_matrix)


def load_data(data=None, is_zero_pad=False):
    """Load data based on the specified dataset."""
    if data == "gdsc1":
        print("load gdsc1")
        PATH = "../gdsc1_data/"
        return _load_data(PATH, is_zero_pad=is_zero_pad)
    elif data == "gdsc2":
        print("load gdsc2")
        PATH = "../gdsc2_data/"
        return _load_data(PATH, is_zero_pad=is_zero_pad)
    elif data == "ctrp":
        PATH = "../ctrp_data/"
        return _load_data(PATH, is_ctrp=True, is_zero_pad=is_zero_pad)
    elif data == "nci":
        print("load nci")
        PATH = "../nci_data/"
        return _load_nci(PATH, is_zero_pad=is_zero_pad)
    else:
        raise NotImplementedError


def _get_base_data(PATH):
    """Load and prepare base data common to all datasets."""
    # Load original drug response data
    drugAct = pd.read_csv(PATH + "drugAct.csv", index_col=0)

    # Load and concatenate gene expression data
    exprs = pd.concat(
        [
            pd.read_csv(PATH + "gene_exp_part1.csv.gz", index_col=0),
            pd.read_csv(PATH + "gene_exp_part2.csv.gz", index_col=0),
        ]
    ).T.dropna()

    return drugAct, exprs


def _process_gene_expression(exprs, dti, genes=None):
    """Process gene expression data and select relevant genes."""
    # Calculate gene variance
    variance = exprs.std()
    variance = variance.sort_values(ascending=False)
    variance = pd.DataFrame(variance > np.percentile(variance, 90))
    variance = list(variance[variance[0]].index)

    print("DTI unique genes: ", len(set(dti["Gene"])))
    print("Top 90% variable genes: ", len(variance))
    print("Total: ", len(set(variance) | (set(dti["Gene"]))))

    # Select genes based on variance and DTI data
    genes = sorted(list(set(variance) | (set(dti["Gene"]))))
    exprs = exprs[genes]
    exprs.columns = list(exprs.columns)

    print("Final gene exp shape:", exprs.shape)

    return exprs


def _get_gene_similarity(PATH, gene_norm_cell):
    """Get or compute gene similarity matrix."""
    gene_sim_files = glob.glob(PATH + "gene_sim/gene_sim_part_*.parquet")

    if gene_sim_files:
        file_paths = glob.glob(f"{PATH}gene_sim/gene_sim_part_*.parquet")
        sorted_file_paths = sorted(file_paths, key=natural_sort_key)

        gene_sim = pd.concat(
            [pd.read_parquet(file) for file in tqdm(sorted_file_paths)]
        )
    else:
        gene_sim = normalize_similarity_matrix(gene_norm_cell.T)
        os.makedirs(PATH + "gene_sim", exist_ok=True)
        chunks = np.array_split(gene_sim, 25)
        for i, chunk in tqdm(enumerate(chunks)):
            chunk.to_parquet(
                f"{PATH}gene_sim/gene_sim_part_{i}.parquet", compression="gzip"
            )

    return gene_sim


def _get_cell_similarity(PATH, gene_norm_gene):
    """Get or compute cell similarity matrix."""
    cell_sim_file = PATH + "cell_sim.csv"
    if os.path.exists(cell_sim_file):
        cell_sim = pd.read_csv(cell_sim_file, index_col=0)
    else:
        cell_sim = normalize_similarity_matrix(gene_norm_gene)
        cell_sim.to_csv(cell_sim_file)

    return cell_sim


def _get_drug_features(PATH, drugAct, smiles_data, smiles_key="SMILES", drug_key=None):
    """Get or compute drug features."""
    try:
        # Attempt to load precomputed drug features
        drug_feature = pd.read_csv(PATH + "drug_feature.csv", index_col=0, header=0)
    except FileNotFoundError:
        # If not found, compute Morgan fingerprints from SMILES
        if isinstance(smiles_data, dict):
            conv = smiles_data
        else:
            conv = dict(smiles_data[[drug_key, smiles_key]].values)

        if drug_key:
            SMILES = [conv[i] for i in drugAct.index]
            drug_feature = get_morgan_fingerprint(SMILES)
            drug_feature_df = pd.DataFrame(drug_feature, index=drugAct.index)
        else:
            drug_feature = get_morgan_fingerprint([conv[i] for i in drugAct.index])
            drug_feature_df = pd.DataFrame(drug_feature, drugAct.index)

        drug_feature_df.to_csv(PATH + "drug_feature.csv")
        drug_feature = drug_feature_df

    return drug_feature


def _get_normalized_gene_data(exprs):
    """Normalize gene expression data."""
    gene_norm_cell = pd.DataFrame(
        StandardScaler().fit_transform(exprs),
        index=exprs.index,
        columns=exprs.columns,
    )

    gene_norm_gene = pd.DataFrame(
        StandardScaler().fit_transform(exprs.T),
        index=exprs.columns,
        columns=exprs.index,
    ).T

    return gene_norm_cell, gene_norm_gene


def _load_data(PATH, is_ctrp=False, is_zero_pad=False):
    """Load and process GDSC1 dataset."""
    # Load original drug response data
    drugAct, exprs = _get_base_data(PATH)

    cells = sorted(
        set(drugAct.columns)
        & set(exprs.index)
        & set(pd.read_csv(PATH + "mut.csv", index_col=0).T.index)
    )

    SMILES = pd.read_csv(PATH + "drug2smiles.csv", index_col=0)
    exprs = exprs.loc[cells]
    drugAct = drugAct.loc[sorted(SMILES.drugs), cells]

    if is_ctrp:
        drugAct = drugAct.apply(lambda x: (x - np.nanmean(x)) / np.nanstd(x))

    # Convert drug activity to binary response matrix
    res = (drugAct > 0).astype(int)
    pos_num = sp.coo_matrix(res).data.shape[0]

    # Get drug features and similarity
    drug_feature = _get_drug_features(PATH, drugAct, SMILES, "SMILES", "drugs")
    drug_sim = normalize_similarity_matrix(drug_feature)

    # Select genes which are top 10% variance and included in DTI dataset
    dti = pd.read_csv("../data/full_table.csv")
    dti = dti[dti["Drug Name"].isin(drugAct.index)]
    dti = dti[dti.Gene.isin(set(exprs.columns) & set(dti.Gene))]

    print("unique drugs:", len(set(dti["Drug Name"])))
    print("unique genes:", len(set(dti.Gene)))

    # Process gene expression data
    exprs = _process_gene_expression(exprs, dti)
    print("Final drug Act shape:", drugAct.shape)

    # Normalize gene data
    gene_norm_cell, gene_norm_gene = _get_normalized_gene_data(exprs)

    # Get gene and cell similarity matrices
    gene_sim = _get_gene_similarity(PATH, gene_norm_cell)
    cell_sim = _get_cell_similarity(PATH, gene_norm_gene)

    # Create adjacency matrices
    A_cg = gene_norm_cell * (gene_norm_cell > 0).astype(int)

    if is_zero_pad:
        A_dg = (
            pd.DataFrame(
                np.zeros([len(drugAct.index), len(A_cg.columns)]),
                index=drugAct.index,
                columns=A_cg.columns,
            )
        )
    else:
        A_dg = (
            pd.DataFrame(
                np.ones([len(drugAct.index), len(A_cg.columns)]),
                index=drugAct.index,
                columns=A_cg.columns,
            )
            / 2
        )
        
    for _, i in dti.iterrows():
        A_dg.loc[i["Drug Name"], i["Gene"]] = 1

    # Create null mask for missing drug activity data
    null_mask = (drugAct.isna()).astype(int)

    # Convert to tensors
    drug_sim = torch.tensor(drug_sim.values).float()
    cell_sim = torch.tensor(cell_sim.values).float()
    gene_sim = torch.tensor(gene_sim.values).float()

    drug, cell, gene = drug_feature, gene_norm_gene, gene_norm_cell
    drug = torch.tensor(drug.values).float()
    cell = torch.tensor(cell.values).float()
    gene = torch.tensor(gene.values).float()

    print("Done!")
    return (
        res,
        pos_num,
        null_mask,
        drug_sim,
        cell_sim,
        gene_sim,
        A_cg,
        A_dg,
        drug,
        cell,
        gene,
    )


def _load_nci(PATH, is_zero_pad):
    """Load and process NCI dataset."""
    # Load original drug response data
    drugAct, exprs = _get_base_data(PATH)
    drugAct.columns = exprs.index
    cells = sorted(
        set(drugAct.columns)
        & set(exprs.index)
        & set(pd.read_csv(PATH + "mut.csv", index_col=0).T.index)
    )

    # Load mechanism of action (moa) data
    moa = pd.read_csv("../Figs/nsc_cid_smiles_class_name.csv", index_col=0)

    # Filter drugs that have SMILES information
    drugAct = drugAct[drugAct.index.isin(moa.NSC)]

    # Load drug synonyms and filter based on availability in other datasets
    tmp = pd.read_csv("../data/drugSynonym.csv")
    tmp = tmp[
        (~tmp.nci60.isna() & ~tmp.ctrp.isna())
        | (~tmp.nci60.isna() & ~tmp.gdsc1.isna())
        | (~tmp.nci60.isna() & ~tmp.gdsc2.isna())
    ]
    tmp = [int(i) for i in set(tmp["nci60"].str.split("|").explode())]

    # Select drugs not classified as 'Other' in MOA and included in other datasets
    drugAct = drugAct.loc[
        sorted(
            set(drugAct.index)
            & (set(moa[moa["MECHANISM"] != "Other"]["NSC"]) | set(tmp))
        )
    ]

    exprs = exprs.loc[cells]
    drugAct = drugAct.loc[:, cells]
    # exprs = np.array(exprs, dtype=np.float32)

    # Convert drug activity to binary response matrix
    res = (drugAct > 0).astype(int)
    pos_num = sp.coo_matrix(res).data.shape[0]

    # Get drug features and similarity
    drug_feature = _get_drug_features(PATH, drugAct, moa, "SMILES", "NSC")
    drug_sim = normalize_similarity_matrix(drug_feature)

    # Select genes which are top 10% variance and included in DTI dataset
    dti = pd.read_csv("../data/full_table.csv")
    dti = dti.dropna(subset="NSC").reset_index(drop=True)
    dti["NSC"] = dti["NSC"].astype(int)
    dti = dti[dti["NSC"].isin(drugAct.index)]
    dti = dti[dti.Gene.isin(set(exprs.columns) & set(dti.Gene))]

    print("unique drugs:", len(set(dti["Drug Name"])))
    print("unique genes:", len(set(dti.Gene)))

    # Process gene expression data
    exprs = _process_gene_expression(exprs, dti)
    print("Final drug Act shape:", drugAct.shape)

    # Normalize gene data
    gene_norm_cell, gene_norm_gene = _get_normalized_gene_data(exprs)

    # Get gene and cell similarity matrices
    gene_sim = _get_gene_similarity(PATH, gene_norm_cell)
    cell_sim = _get_cell_similarity(PATH, gene_norm_gene)

    # Create adjacency matrices
    A_cg = min_max_scale(gene_norm_gene + gene_norm_cell)

    if is_zero_pad:
        A_dg = (
            pd.DataFrame(
                np.zeros([len(drugAct.index), len(A_cg.columns)]),
                index=drugAct.index,
                columns=A_cg.columns,
            )
        )
    else:
        A_dg = (
            pd.DataFrame(
                np.ones([len(drugAct.index), len(A_cg.columns)]),
                index=drugAct.index,
                columns=A_cg.columns,
            )
            / 2
        )
        
    for _, i in dti.iterrows():
        A_dg.loc[int(i["NSC"]), i["Gene"]] = 1

    # Create null mask for missing drug activity data
    null_mask = (drugAct.isna()).astype(int)

    # Convert to tensors
    drug_sim = torch.tensor(drug_sim.values).float()
    cell_sim = torch.tensor(cell_sim.values).float()
    gene_sim = torch.tensor(gene_sim.values).float()

    drug, cell, gene = drug_feature, gene_norm_gene, gene_norm_cell.T
    drug = torch.tensor(drug.values).float()
    cell = torch.tensor(cell.values).float()
    gene = torch.tensor(gene.values).float()

    print("Done!")
    return (
        res,
        pos_num,
        null_mask,
        drug_sim,
        cell_sim,
        gene_sim,
        A_cg,
        A_dg,
        drug,
        cell,
        gene,
    )
