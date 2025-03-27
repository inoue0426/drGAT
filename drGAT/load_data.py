import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from .utility import (
    get_morgan_fingerprint,
    min_max_scale,
    natural_sort_key,
    normalize_similarity_matrix,
)


def load_data(data=None):
    # Load data based on the specified dataset
    # if data == "gdsc1":
    #     print("load gdsc1")
    #     return _load_gdsc1()
    # elif data == "gdsc2":
    #     print("load gdsc2")
    #     return _load_gdsc2()
    # elif data == "ctrp":
    #     print("load ctrp")
    #     return _load_ctrp()
    # else:
    print("load nci")
    PATH = "../nci_data/"
    return _load_nci(PATH)


def _load_nci(PATH="../nci_data/"):
    # Load original drug response data
    drugAct = pd.read_csv(PATH + "drugAct.csv", index_col=0)

    # Load and concatenate gene expression data
    exprs = pd.concat(
        [
            pd.read_csv(PATH + "gene_exp_part1.csv.gz", index_col=0),
            pd.read_csv(PATH + "gene_exp_part2.csv.gz", index_col=0),
        ]
    ).T

    drugAct.columns = exprs.index

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

    # Convert drug activity to binary response matrix
    res = (drugAct > 0).astype(int)
    pos_num = sp.coo_matrix(res).data.shape[0]

    try:
        # Attempt to load precomputed drug features
        drug_feature = pd.read_csv(PATH + "drug_feature.csv", index_col=0, header=0)
    except FileNotFoundError:
        # If not found, compute Morgan fingerprints from SMILES
        conv = dict(moa[["NSC", "SMILES"]].values)
        SMILES = [conv[i] for i in drugAct.index]
        drug_feature = get_morgan_fingerprint(SMILES)
        drug_feature_df = pd.DataFrame(drug_feature, index=drugAct.index)
        drug_feature_df.to_csv(PATH + "drug_feature.csv")

    drug_sim = normalize_similarity_matrix(drug_feature)

    # Select genes which are top 10 % variance and included in DTI dataset
    dti = pd.read_csv("../data/full_table.csv")
    dti = dti.dropna(subset="NSC").reset_index(drop=True)
    dti["NSC"] = dti["NSC"].astype(int)
    dti = dti[dti["NSC"].isin(drugAct.index)]
    dti = dti[dti.Gene.isin(set(exprs.columns) & set(dti.Gene))]

    variance = exprs.std()
    variance = variance.sort_values(ascending=False)
    variance = pd.DataFrame(variance > np.percentile(variance, 90))
    variance = list(variance[variance[0]].index)

    genes = sorted(list(set(variance) | (set(dti["Gene"]))))
    exprs = exprs[genes]
    exprs.columns = list(exprs.columns)

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

    gene_sim_files = glob.glob(PATH + "gene_sim/gene_sim_part_*.parquet")

    if gene_sim_files:
        file_paths = glob.glob("../nci_data/gene_sim/gene_sim_part_*.parquet")
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

    cell_sim_file = PATH + "cell_sim.csv"
    if os.path.exists(cell_sim_file):
        cell_sim = pd.read_csv(cell_sim_file, index_col=0)
    else:
        cell_sim = normalize_similarity_matrix(gene_norm_gene)
        cell_sim.to_csv(cell_sim_file)

    A_cg = min_max_scale(gene_norm_gene + gene_norm_cell)

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

    drug_sim = torch.tensor(drug_sim.values).float()
    cell_sim = torch.tensor(cell_sim.values).float()
    gene_sim = torch.tensor(gene_sim.values).float()

    print("Done!")
    return res, pos_num, null_mask, drug_sim, cell_sim, gene_sim, A_cg, A_dg
