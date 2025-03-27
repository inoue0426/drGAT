import glob
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# How to read
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", s)]


def reconstruct_drug_sim():
    file_paths = glob.glob("../nci_data/drug_sim/drug_sim_part_*.parquet")
    sorted_file_paths = sorted(file_paths, key=natural_sort_key)
    return pd.concat([pd.read_parquet(file) for file in tqdm(sorted_file_paths)])


def load_and_combine_chunks(pattern, axis=0):
    chunk_files = sorted(
        glob.glob(pattern), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    chunks = [np.load(f) for f in chunk_files]
    return np.concatenate(chunks, axis=axis)


def get_morgan_fingerprint(SMILES):
    # Initialize parser parameters
    params = Chem.SmilesParserParams()
    params.useChirality = True  # Preserve stereochemistry information
    params.removeHs = False  # Keep hydrogen atoms
    mfp = []

    for smi in SMILES:
        mol = None
        # Sanitization attempt strategies
        sanitize_attempts = [
            {"sanitize": True},  # First try with standard sanitization
            {
                "sanitize": False,
                "partial_sanitize": True,
            },  # Fallback: partial sanitization
        ]

        for attempt in sanitize_attempts:
            try:
                # Update parameters for this attempt
                current_params = Chem.SmilesParserParams()
                current_params.sanitize = attempt["sanitize"]
                current_params.useChirality = params.useChirality
                current_params.removeHs = params.removeHs

                # Molecule object creation
                mol = Chem.MolFromSmiles(smi, params=current_params)

                if mol and "partial_sanitize" in attempt:
                    # Perform partial sanitization (skip property validation)
                    Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)

                if mol:  # Successfully processed molecule
                    # Generate Morgan fingerprint
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    mfp.append(np.array(fp))
                    break  # Exit attempt loop on success

            except Exception as e:
                if attempt == sanitize_attempts[-1]:  # Final attempt failed
                    print(f"Processing failed: {smi}")
                    print(f"Error details: {str(e)}")
                continue  # Try next attempt

        if not mol:  # All attempts failed
            print(f"Complete processing failure: {smi}")
            mfp.append(np.zeros(2048))  # Insert zero-vector placeholder

    return np.array(mfp)


def min_max_scale(data):
    data = data[data > 0].fillna(0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    return pd.DataFrame(
        scaler.fit_transform(data), index=data.index, columns=data.columns
    )


def normalize_similarity_matrix(df, gamma=None):
    similarity_matrix = rbf_kernel(df.values, gamma=gamma)
    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(similarity_matrix.reshape(-1, 1))
    normalized_df = pd.DataFrame(
        normalized_matrix.reshape(similarity_matrix.shape),
        index=df.index,
        columns=df.index,
    )

    return normalized_df
