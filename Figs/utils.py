import ast
import json
import os
import pickle
import time
from collections import defaultdict
from typing import Any, List

import autogen
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests  # type: ignore
import seaborn as sns
from Bio import Entrez
from DeepPurpose.dataset import load_broad_repurposing_hub
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from tqdm import tqdm

# Set your email for Entrez
Entrez.email = "your_email@example.com"

# --- Simple persistent cache for PubMed hit counts ---
_CACHE_FILE = "pubmed_hit_count_cache.pkl"
if os.path.exists(_CACHE_FILE):
    with open(_CACHE_FILE, "rb") as f:
        _PUBMED_HIT_COUNT_CACHE = pickle.load(f)
else:
    _PUBMED_HIT_COUNT_CACHE = {}


def save_pubmed_hit_count_cache():
    with open(_CACHE_FILE, "wb") as f:
        pickle.dump(_PUBMED_HIT_COUNT_CACHE, f)


def get_pubmed_hit_count(drug, gene, max_retries=5, delay=1.0):
    key = (str(drug), str(gene))
    if key in _PUBMED_HIT_COUNT_CACHE:
        return _PUBMED_HIT_COUNT_CACHE[key]
    query = f"{drug} AND {gene}"
    for attempt in range(max_retries):
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            record = Entrez.read(handle)
            handle.close()
            count = int(record["Count"])
            _PUBMED_HIT_COUNT_CACHE[key] = count
            save_pubmed_hit_count_cache()
            return count
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                _PUBMED_HIT_COUNT_CACHE[key] = -1
                save_pubmed_hit_count_cache()
                return -1
        except Exception:
            _PUBMED_HIT_COUNT_CACHE[key] = -1
            save_pubmed_hit_count_cache()
            return -1
    _PUBMED_HIT_COUNT_CACHE[key] = -1
    save_pubmed_hit_count_cache()
    return -1  # Failed after retries


def plot_heatmap(
    df_moa,
    dti_path="nci_dti.csv",
    font_size=14,
    output_path="fig2.pdf",
    MOA_name="Kinase",
    specific_drug=None,
    figsize=(30, 15),
    dpi=300,
    font_family="Arial",
    legend_loc=(0.5, 0.155),
):
    """
    Plot a predicted Drug-Gene heatmap for kinase inhibitors.
    """
    # --- Figure/Font settings ---
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rcParams["font.family"] = font_family
    plt.rcParams["pdf.fonttype"] = 42

    # --- Data loading and preprocessing ---
    df = df_moa[df_moa["MECHANISM"] == MOA_name].copy()
    if specific_drug:
        df = df[df["drug"].str.lower().str.contains(specific_drug)]
    df["drug"] = df["drug"].replace("Erlotinib hydrochloride", "Erlotinib")

    dti = pd.read_csv(dti_path, index_col=0)[["NSC", "Drug Name", "Gene"]]
    dti.columns = ["NSC", "drug", "gene"]
    dti["NSC"] = dti["NSC"].astype(int)

    dti["key"] = dti["NSC"].astype(str) + "_" + dti["gene"]
    df["key"] = df["NSC"].astype(str) + "_" + df["gene"]
    df["dti_existed"] = df["key"].isin(set(dti["key"])).astype(int)

    # --- Pivot for heatmap ---
    pivot = df.pivot_table(index="drug", columns="gene", values="num", aggfunc="sum")
    pivot = pivot.loc[(pivot.sum(axis=1) > 0), (pivot.sum() > 0)]
    log_pivot = np.log1p(pivot)

    # --- Marker categories (based on model prediction vs DTI existence) ---
    predicted_df = df[df["rank"] <= 5]
    predicted_keys = set(predicted_df["key"])
    dti_keys = set(dti["key"])
    tp_keys = predicted_keys & dti_keys
    fp_keys = predicted_keys - dti_keys
    fn_keys = dti_keys - predicted_keys

    drug_to_nscs = defaultdict(set)
    for _, row in df[["drug", "NSC"]].drop_duplicates().iterrows():
        drug_to_nscs[row["drug"]].add(str(row["NSC"]))

    # Generate pivot_keys for all drug-gene pairs in the heatmap
    pivot_keys = {
        f"{nsc}_{gene}"
        for drug in pivot.index
        for gene in pivot.columns
        for nsc in drug_to_nscs.get(drug, [])
    }

    # --- True Positives (TP) ---
    tp = predicted_df[predicted_df["key"].isin(tp_keys & pivot_keys)].drop_duplicates(
        "key"
    )

    # --- False Positives (FP) ---
    fp = predicted_df[predicted_df["key"].isin(fp_keys & pivot_keys)].drop_duplicates(
        "key"
    )

    # --- False Negatives (FN) ---
    fn_df = df[df["key"].isin(fn_keys & pivot_keys)].drop_duplicates("key")
    fn_missing = dti[dti["key"].isin((fn_keys - set(df["key"])) & pivot_keys)].copy()

    results = []
    if len(fn_missing) > 0:
        for drug, gene in tqdm(zip(fn_missing["drug"], fn_missing["gene"])):
            count = get_pubmed_hit_count(drug, gene)
            results.append(count)
            time.sleep(0.34)  # NCBI recommendation: up to 3 requests per second
        fn_missing["num"] = results
    else:
        fn_missing["num"] = []

    fn = pd.concat([fn_df, fn_missing], ignore_index=True).drop_duplicates("key")

    # --- Plotting ---
    base_cmap = plt.get_cmap("YlOrRd", 256)
    colors = base_cmap(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    new_cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(14, 20))
    heatmap = sns.heatmap(
        log_pivot,
        cmap=new_cmap,
        ax=ax,
        square=True,
        linewidths=0.1,
        cbar_kws={"label": "Number of Papers (log)", "shrink": 0.2, "pad": 0.005},
        linecolor="lightgray",
    )

    def annotate_markers(df_mark, marker, color, size=10):
        for _, row in df_mark.iterrows():
            try:
                y = pivot.index.get_loc(row["drug"])
                x = pivot.columns.get_loc(row["gene"])
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    marker,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=size,
                )
            except KeyError:
                continue

    annotate_markers(tp, "♥", "black", size=13)
    annotate_markers(fp, "♣", "black", size=13)
    annotate_markers(fn, "♦", "black", size=15)

    # --- Number of PubMed-supported cases for each marker type ---
    tp_with_paper = (tp["num"] > 0).sum() if "num" in tp else 0
    fp_with_paper = (fp["num"] > 0).sum() if "num" in fp else 0
    fn_with_paper = (fn["num"] > 0).sum() if "num" in fn else 0

    # --- Count for each marker type ---
    tp_count = len(tp)
    fp_count = len(fp)
    fn_count = len(fn)

    # --- Create horizontal legend text ---
    legend_labels = [
        f"♥: Predicted & in DTI ({tp_with_paper}/{tp_count} with papers)",
        f"♣: Predicted Only ({fp_with_paper}/{fp_count} with papers)",
        f"♦: Only in DTI ({fn_with_paper}/{fn_count} with papers)",
    ]
    legend_text = "   |   ".join(legend_labels)
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)

    fig.text(
        legend_loc[0],
        legend_loc[1],
        legend_text,
        ha="center",
        va="top",
        fontsize=font_size,
        bbox=props,
    )

    ax.set_xlabel("Gene", fontsize=font_size)
    ax.set_ylabel("Drug", fontsize=font_size)
    ax.set_title(
        "Top-5 Predicted Drug–Gene Interactions for Kinase Inhibitors vs. Known DTI",
        fontsize=font_size,
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=font_size, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size, rotation=0)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(font_size)
    cbar.ax.tick_params(labelsize=font_size)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_drug_gene_network_and_barplot(
    df,
    target_drugs,
    gea,
    output_path="drug_gene_network.pdf",
    font_family="Arial",
    font_size=6,
    figsize=(7, 4),
    dpi=300,
    width_ratios=[3, 1],
    min_term_count=10,
):
    drugs = df[df.drug.isin(target_drugs)].copy()
    drugs = drugs[drugs["NSC"] != 715055]
    drugs["drug"] = drugs["drug"].replace("Erlotinib hydrochloride", "Erlotinib")

    plt.rcParams["font.family"] = font_family
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = font_size

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": width_ratios}
    )

    # --- Network plot ---
    G = nx.from_pandas_edgelist(
        drugs,
        source="gene",
        target="drug",
        edge_attr=["atten_score", "dti_existed", "num"],
    )

    pos = nx.kamada_kawai_layout(G, scale=1)

    genes = set(drugs["gene"])
    node_colors = ["green" if node in genes else "orange" for node in G.nodes()]

    edges = G.edges(data=True)
    atten_scores = [attr["atten_score"] for _, _, attr in edges]
    norm = plt.Normalize(min(atten_scores), max(atten_scores))
    cmap = plt.cm.YlOrRd
    cmap_mod = plt.cm.colors.ListedColormap(cmap(np.linspace(0.3, 1.0, 256)))

    # --- Label position adjustment ---
    preset_target_drugs = [
        "Gefitinib",
        "Vemurafenib",
        "Nilotinib",
        "Dovitinib",
        "Erlotinib hydrochloride",
    ]

    preset_target_drugs_2 = [
        "Topotecan",
        "Camptothecin",
        "Irinotecan",
        "Adriblastin",
        "Daunorubicin",
    ]

    if sorted(target_drugs) == sorted(preset_target_drugs):
        target_drug = "Nilotinib"
        offset_y = 0.0

        if target_drug in pos:
            pos[target_drug] = (pos[target_drug][0], pos[target_drug][1] + offset_y)
            for neighbor in G.neighbors(target_drug):
                pos[neighbor] = (pos[neighbor][0], pos[neighbor][1] + offset_y)

        def safe_shift(node, dx=0, dy=0):
            if node in pos:
                pos[node] = (pos[node][0] + dx, pos[node][1] + dy)

        safe_shift("EGFR", 0, 0.3)
        safe_shift("PRKAG2", 0, 0.5)
        safe_shift("ENDOD1", 0, -0.5)
        safe_shift("STK17B", 0, -0.1)
        safe_shift("PIK3C3", 0, -0.1)

        label_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}

        def safe_label_shift(node, dx=0, dy=0):
            if node in label_pos:
                label_pos[node] = (label_pos[node][0] + dx, label_pos[node][1] + dy)

        safe_label_shift("Gefitinib", -0.15, 0)
        safe_label_shift("Vemurafenib", 0.1, 0.06)
        safe_label_shift("NRN1", -0.12, 0)
        safe_label_shift("ZNF667-AS1", 0.2, 0)
        safe_label_shift("Erlotinib", -0.15, 0)
        safe_label_shift("Dovitinib", 0.15, 0)
        safe_label_shift("BRAF", -0.12, 0)
        safe_label_shift("SLC43A3", 0.15, 0)
        safe_label_shift("DDR2", 0, -0.05)
        safe_label_shift("STK17B", 0, -0.1)
        safe_label_shift("HIST1H3I", -0.1, -0.1)
        safe_label_shift("HIST1H2BB", 0, -0.1)
    elif sorted(target_drugs) == sorted(preset_target_drugs_2):
        # Topotecan
        target_drug = "Topotecan"
        offset_x = 0
        offset_y = 0.1
        if target_drug in pos:
            pos[target_drug] = (
                pos[target_drug][0] + offset_x,
                pos[target_drug][1] + offset_y,
            )
            for neighbor in G.neighbors(target_drug):
                pos[neighbor] = (
                    pos[neighbor][0] + offset_x,
                    pos[neighbor][1] + offset_y,
                )

        # Camptothecin
        target_drug = "Camptothecin"
        offset_x = 0.5
        offset_y = 0.7
        if target_drug in pos:
            pos[target_drug] = (
                pos[target_drug][0] + offset_x,
                pos[target_drug][1] + offset_y,
            )
            for neighbor in G.neighbors(target_drug):
                pos[neighbor] = (
                    pos[neighbor][0] + offset_x,
                    pos[neighbor][1] + offset_y,
                )

        # Irinotecan
        target_drug = "Irinotecan"
        offset_x = 0.3
        offset_y = 0.3
        if target_drug in pos:
            pos[target_drug] = (
                pos[target_drug][0] + offset_x,
                pos[target_drug][1] + offset_y,
            )
            for neighbor in G.neighbors(target_drug):
                pos[neighbor] = (
                    pos[neighbor][0] + offset_x,
                    pos[neighbor][1] + offset_y,
                )

        # Adriblastin
        target_drug = "Adriblastin"
        offset_x = 0.2
        offset_y = -0.2
        if target_drug in pos:
            pos[target_drug] = (
                pos[target_drug][0] + offset_x,
                pos[target_drug][1] + offset_y,
            )
            for neighbor in G.neighbors(target_drug):
                pos[neighbor] = (
                    pos[neighbor][0] + offset_x,
                    pos[neighbor][1] + offset_y,
                )

        # Daunorubicin
        target_drug = "Daunorubicin"
        offset_x = 0.5
        offset_y = 0
        if target_drug in pos:
            pos[target_drug] = (
                pos[target_drug][0] + offset_x,
                pos[target_drug][1] + offset_y,
            )
            for neighbor in G.neighbors(target_drug):
                pos[neighbor] = (
                    pos[neighbor][0] + offset_x,
                    pos[neighbor][1] + offset_y,
                )

        # Additional node adjustments
        if "BAX" in pos:
            pos["BAX"] = (pos["BAX"][0], pos["BAX"][1] + 0.1)
        if "HMGCR" in pos:
            pos["HMGCR"] = (pos["HMGCR"][0] - 0.3, pos["HMGCR"][1] - 0.3)
        if "LOC100128816" in pos:
            pos["LOC100128816"] = (
                pos["LOC100128816"][0] + 0.2,
                pos["LOC100128816"][1] - 0.1,
            )
        label_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}

    else:
        label_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=70, ax=ax1)

    for u, v, attr in edges:
        color = cmap_mod(norm(attr["atten_score"]))
        if attr["dti_existed"] == 1:
            style = "-"
        elif attr["num"] > 0:
            style = "--"
        else:
            style = ":"
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color=[color], style=style, width=1, ax=ax1
        )

    nx.draw_networkx_labels(G, label_pos, font_size=font_size, ax=ax1)

    sm = plt.cm.ScalarMappable(cmap=cmap_mod, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.6)
    cbar.set_label("Attention Score", fontsize=font_size, labelpad=-10)
    cbar.set_ticks([min(atten_scores), max(atten_scores)])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=font_size)

    green_patch = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="green",
        markersize=font_size,
        label="Gene",
    )
    orange_patch = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="orange",
        markersize=font_size,
        label="Drug",
    )

    ax1.legend(
        handles=[
            green_patch,
            orange_patch,
            Line2D(
                [0],
                [0],
                color="gray",
                lw=1,
                ls="-",
                label="Known Drug-Target Interaction",
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                lw=1,
                ls="--",
                label="Known Drug-Gene Association",
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                lw=1,
                ls=":",
                label="Predicted Drug-Gene Association",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0, 0),
    )

    ax1.set_title(
        "Drug-Gene Network with Attention Scores and Known Interactions",
        fontsize=font_size,
    )
    ax1.axis("off")

    # --- Barplot ---
    color_list = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#9a6324",
        "#fffac8",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000075",
        "#808080",
        "#ffffff",
        "#000000",
    ]

    agg_df = (
        gea[["Term", "drug", "moa"]]
        .drop_duplicates()
        .groupby(["Term", "moa"])["drug"]
        .count()
        .reset_index(name="Drug Count")
    )

    pivot_df = agg_df.pivot(index="Term", columns="moa", values="Drug Count").fillna(0)
    pivot_df = pivot_df.loc[pivot_df.sum(axis=1) >= min_term_count]
    pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]
    pivot_df = pivot_df[::-1]

    columns = list(pivot_df.columns)
    if "Other" in columns:
        columns.remove("Other")
        columns.append("Other")
    pivot_df = pivot_df[columns]

    color_map = {
        col: color_list[i % len(color_list)] for i, col in enumerate(pivot_df.columns)
    }

    bottom = np.zeros(len(pivot_df))
    for col in pivot_df.columns:
        ax2.barh(
            y=pivot_df.index,
            width=pivot_df[col],
            left=bottom,
            label=col,
            color=color_map[col],
        )
        bottom += pivot_df[col].values

    ax2.set_xlabel("# of Drugs")
    ax2.set_ylabel("Biological Process")
    ax2.set_title(
        "Drug-Associated Biological Processes Grouped by Mechanism of Action",
        fontsize=font_size,
        loc="left",
        x=-1,
    )

    ax2.legend(
        title="Mechanism",
        bbox_to_anchor=(0.4, 0.5),
        loc="upper left",
        fontsize=font_size,
    )

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.show()


def get_smiles_from_compound_name(compound_name):
    # First try to get SMILES from local file
    try:
        df = pd.read_csv(
            "../data/nsc_cid_smiles_class_name.csv", usecols=["NAME", "SMILES"]
        )
        # Look for matching compound name
        match = df[df["NAME"].str.lower() == compound_name.lower()]
        if not match.empty:
            return match.iloc[0]["SMILES"]
    except Exception as e:
        print(f"Error reading local SMILES data: {e}")

    # If not found locally, try PubChem API
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/CanonicalSMILES/JSON"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        smiles = (
            data.get("PropertyTable", {})
            .get("Properties", [{}])[0]
            .get("CanonicalSMILES")
        )
        return smiles
    except requests.RequestException as e:
        print(f"Error retrieving SMILES for compound name {compound_name}: {e}")
        return None


def get_sequence_from_target_name(target_name):
    url = (
        f"https://rest.uniprot.org/uniprotkb/search?query={target_name}&fields=sequence"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [{}])[0].get("sequence", {}).get("value")
    except requests.RequestException as e:
        print(f"Error retrieving sequence for target name {target_name}: {e}")
        return None
