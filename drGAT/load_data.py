from pathlib import Path
import glob
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

from pathlib import Path

# カレントディレクトリの親をたどって "drGAT" ディレクトリを検出し、その1つ上をルートとする
current = Path.cwd().resolve()
while current.name != "drGAT":
    if current.parent == current:
        raise RuntimeError("drGAT ディレクトリが見つかりませんでした")
    current = current.parent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGS_DIR = PROJECT_ROOT / "Figs"


def load_data(data=None, is_zero_pad=False):
    if data == "gdsc1":
        print("load gdsc1")
        path = DATA_DIR / "gdsc1_data"
        return _load_data(path, is_zero_pad=is_zero_pad)
    elif data == "gdsc2":
        print("load gdsc2")
        path = DATA_DIR / "gdsc2_data"
        return _load_data(path, is_zero_pad=is_zero_pad)
    elif data == "ctrp":
        path = DATA_DIR / "ctrp_data"
        return _load_data(path, is_ctrp=True, is_zero_pad=is_zero_pad)
    elif data == "nci":
        print("load nci")
        path = DATA_DIR / "nci_data"
        return _load_nci(path, is_zero_pad=is_zero_pad)
    else:
        raise NotImplementedError


def _get_base_data(path):
    drugAct = pd.read_csv(path / "drugAct.csv", index_col=0)
    exprs = pd.concat([
        pd.read_csv(path / "gene_exp_part1.csv.gz", index_col=0),
        pd.read_csv(path / "gene_exp_part2.csv.gz", index_col=0),
    ]).T.dropna()
    return drugAct, exprs


def _process_gene_expression(exprs, dti):
    variance = exprs.std().sort_values(ascending=False)
    variance = pd.DataFrame(variance > np.percentile(variance, 90))
    variance = list(variance[variance[0]].index)

    print("DTI unique genes: ", len(set(dti["Gene"])))
    print("Top 90% variable genes: ", len(variance))
    print('overwrapped genes: ', len(set(dti['Gene']) & set(variance)))
    print("Total: ", len(set(variance) | set(dti["Gene"])))

    genes = sorted(list(set(variance) | set(dti["Gene"])))
    exprs = exprs[genes]
    return exprs


def _get_gene_similarity(path, gene_norm_cell):
    sim_path = path / "gene_sim"
    gene_sim_files = sorted(sim_path.glob("gene_sim_part_*.parquet"), key=lambda x: natural_sort_key(str(x)))

    if gene_sim_files:
        gene_sim = pd.concat([pd.read_parquet(f) for f in tqdm(gene_sim_files)])
    else:
        gene_sim = normalize_similarity_matrix(gene_norm_cell.T)
        sim_path.mkdir(exist_ok=True)
        for i, chunk in enumerate(np.array_split(gene_sim, 25)):
            chunk.to_parquet(sim_path / f"gene_sim_part_{i}.parquet", compression="gzip")
    return gene_sim


def _get_cell_similarity(path, gene_norm_gene):
    sim_file = path / "cell_sim.csv"
    if sim_file.exists():
        cell_sim = pd.read_csv(sim_file, index_col=0)
    else:
        cell_sim = normalize_similarity_matrix(gene_norm_gene)
        cell_sim.to_csv(sim_file)
    return cell_sim


def _get_drug_features(path, drugAct, smiles_data, smiles_key="SMILES", drug_key=None):
    feat_file = path / "drug_feature.csv"
    if feat_file.exists():
        drug_feature = pd.read_csv(feat_file, index_col=0)
    else:
        conv = dict(smiles_data[[drug_key, smiles_key]].values) if not isinstance(smiles_data, dict) else smiles_data
        SMILES = [conv[i] for i in drugAct.index]
        drug_feature_df = pd.DataFrame(get_morgan_fingerprint(SMILES), index=drugAct.index)
        drug_feature_df.to_csv(feat_file)
        drug_feature = drug_feature_df
    return drug_feature


def _get_normalized_gene_data(exprs):
    gene_norm_cell = pd.DataFrame(StandardScaler().fit_transform(exprs), index=exprs.index, columns=exprs.columns)
    gene_norm_gene = pd.DataFrame(StandardScaler().fit_transform(exprs.T), index=exprs.columns, columns=exprs.index).T
    return gene_norm_cell, gene_norm_gene


def _load_data(path, is_ctrp=False, is_zero_pad=False):
    drugAct, exprs = _get_base_data(path)
    mut = pd.read_csv(path / "mut.csv", index_col=0).T
    cells = sorted(set(drugAct.columns) & set(exprs.index) & set(mut.index))

    SMILES = pd.read_csv(path / "drug2smiles.csv", index_col=0)
    exprs = exprs.loc[cells]
    drugAct = drugAct.loc[sorted(SMILES.drugs), cells]

    if is_ctrp:
        drugAct = drugAct.apply(lambda x: (x - np.nanmean(x)) / np.nanstd(x))

    res = (drugAct > 0).astype(int)
    pos_num = sp.coo_matrix(res).data.shape[0]

    drug_feature = _get_drug_features(path, drugAct, SMILES, "SMILES", "drugs")
    drug_sim = normalize_similarity_matrix(drug_feature)

    dti = pd.read_csv(DATA_DIR / "full_table.csv")
    dti = dti[dti["Drug Name"].isin(drugAct.index)]
    dti = dti[dti.Gene.isin(set(exprs.columns) & set(dti.Gene))]

    print('dtis: ', len(dti))
    print("unique drugs:", len(set(dti["Drug Name"])))
    print("unique genes:", len(set(dti.Gene)))

    exprs = _process_gene_expression(exprs, dti)
    print("Final drug Act shape:", drugAct.shape)

    gene_norm_cell, gene_norm_gene = _get_normalized_gene_data(exprs)
    gene_sim = _get_gene_similarity(path, gene_norm_cell)
    cell_sim = _get_cell_similarity(path, gene_norm_gene)

    A_cg = gene_norm_cell * (gene_norm_cell > 0).astype(int)

    A_dg = pd.DataFrame(
        np.zeros([len(drugAct.index), len(A_cg.columns)]) if is_zero_pad else np.ones([len(drugAct.index), len(A_cg.columns)]) / 2,
        index=drugAct.index,
        columns=A_cg.columns,
    )

    for _, i in dti.iterrows():
        A_dg.loc[i["Drug Name"], i["Gene"]] = 1

    null_mask = (drugAct.isna()).astype(int)

    drug_sim = torch.tensor(drug_sim.values).float()
    cell_sim = torch.tensor(cell_sim.values).float()
    gene_sim = torch.tensor(gene_sim.values).float()

    drug = torch.tensor(drug_feature.values).float()
    cell = torch.tensor(gene_norm_gene.values).float()
    gene = torch.tensor(gene_norm_cell.values).float()

    print("Done!")
    return res, pos_num, null_mask, drug_sim, cell_sim, gene_sim, A_cg, A_dg, drug, cell, gene


def _load_nci(path, is_zero_pad):
    drugAct, exprs = _get_base_data(path)
    mut = pd.read_csv(path / "mut.csv", index_col=0).T
    cells = sorted(set(drugAct.columns) & set(exprs.index) & set(mut.index))

    moa = pd.read_csv(FIGS_DIR / "nsc_cid_smiles_class_name.csv", index_col=0)

    exprs = exprs.loc[cells]
    drugAct = drugAct.loc[:, cells]

    res = drugAct
    pos_num = sp.coo_matrix(res).data.shape[0]

    drug_feature = _get_drug_features(path, drugAct, moa, "SMILES", "NSC")
    drug_sim = normalize_similarity_matrix(drug_feature)

    dti = pd.read_csv(DATA_DIR / "full_table.csv").dropna(subset=["NSC"]).reset_index(drop=True)
    dti["NSC"] = dti["NSC"].astype(int)
    dti = dti[dti["NSC"].isin(drugAct.index)]
    dti = dti[dti.Gene.isin(set(exprs.columns) & set(dti.Gene))]
    
    print('dtis: ', len(dti))
    print("unique drugs:", len(set(dti["Drug Name"])))
    print("unique genes:", len(set(dti.Gene)))

    exprs = _process_gene_expression(exprs, dti)
    print("Final drug Act shape:", drugAct.shape)

    gene_norm_cell, gene_norm_gene = _get_normalized_gene_data(exprs)
    gene_sim = _get_gene_similarity(path, gene_norm_cell)
    cell_sim = _get_cell_similarity(path, gene_norm_gene)

    A_cg = min_max_scale(gene_norm_gene + gene_norm_cell)

    A_dg = pd.DataFrame(
        np.zeros([len(drugAct.index), len(A_cg.columns)]) if is_zero_pad else np.ones([len(drugAct.index), len(A_cg.columns)]) / 2,
        index=drugAct.index,
        columns=A_cg.columns,
    )

    for _, i in dti.iterrows():
        A_dg.loc[int(i["NSC"]), i["Gene"]] = 1

    null_mask = (drugAct.isna()).astype(int)

    drug_sim = torch.tensor(drug_sim.values).float()
    cell_sim = torch.tensor(cell_sim.values).float()
    gene_sim = torch.tensor(gene_sim.values).float()

    drug = torch.tensor(drug_feature.values).float()
    cell = torch.tensor(gene_norm_gene.values).float()
    gene = torch.tensor(gene_norm_cell.T.values).float()

    print("Done!")
    return res, pos_num, null_mask, drug_sim, cell_sim, gene_sim, A_cg, A_dg, drug, cell, gene
