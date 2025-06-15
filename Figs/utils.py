import ast
import json
import os
from typing import Any, List

import autogen
import numpy as np
import pandas as pd
import requests  # type: ignore
from DeepPurpose.dataset import load_broad_repurposing_hub
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


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
