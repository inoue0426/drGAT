import glob
import re

import numpy as np
import pandas as pd
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
