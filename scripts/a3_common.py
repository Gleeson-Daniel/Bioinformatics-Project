import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

METADATA_PATH = "data/metadata_clean.tsv"
SUBSET_DIR = Path("data/subsets")
CLUSTER_DIR = Path("results/clusters")
CHOSEN_K_PATH = Path("results/a3_chosen_k.json")

GENE_COUNTS = [10, 100, 1000, 5000, 10000]
MAIN_N_GENES = 5000
K_RANGE = range(2, 11)
RANDOM_STATE = 42


def load_metadata():
    """Metadata indexed by sample ID (same IDs as the counts matrix columns)."""
    return pd.read_csv(METADATA_PATH, sep="\t").set_index("refinebio_accession_code")


def load_subset(n_genes):
    """VST values for the top n most variable genes, genes x samples."""
    return pd.read_csv(SUBSET_DIR / f"vst_top{n_genes}.tsv.gz", sep="\t", index_col=0)


def samples_by_genes(n_genes):
    """Same data transposed to samples x genes, the shape scikit-learn expects."""
    return load_subset(n_genes).T


def label_path(method, n_genes, k):
    return CLUSTER_DIR / f"{method}_{n_genes}g_k{k}.csv"


def save_labels(method, n_genes, k, sample_ids, labels):
    """Save one clustering result as sample,cluster (clusters numbered from 1)."""
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"sample": list(sample_ids), "cluster": np.asarray(labels) + 1})
    df.to_csv(label_path(method, n_genes, k), index=False)


def load_labels(method, n_genes, k):
    df = pd.read_csv(label_path(method, n_genes, k))
    return df.set_index("sample")["cluster"]


def save_chosen_k(method, k):
    chosen = load_chosen_k() if CHOSEN_K_PATH.exists() else {}
    chosen[method] = int(k)
    CHOSEN_K_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHOSEN_K_PATH.write_text(json.dumps(chosen, indent=2))


def load_chosen_k():
    return json.loads(CHOSEN_K_PATH.read_text())


def describe_transition(prev_labels, curr_labels, k_prev, k_curr):
    """Crosstab of cluster membership at k_prev vs k_curr, plus whether it nests.

    Nested means every cluster at k_curr sits entirely inside one cluster at
    k_prev, i.e. going up in k only split existing clusters.
    """
    tab = pd.crosstab(
        pd.Series(prev_labels, name=f"k={k_prev}"),
        pd.Series(curr_labels, name=f"k={k_curr}"),
    )
    nested = bool(((tab > 0).sum(axis=0) == 1).all())
    ari = adjusted_rand_score(prev_labels, curr_labels)
    header = f"k={k_prev} -> k={k_curr}   nested={nested}   ARI={ari:.3f}"
    return f"{header}\n{tab.to_string()}\n", nested, ari