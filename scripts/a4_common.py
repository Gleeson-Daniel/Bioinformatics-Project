from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from a3_common import GENE_COUNTS, RANDOM_STATE, load_labels, load_metadata, samples_by_genes

PRED_DIR = Path("results/predictions")
SIG_DIR = Path("results/signatures")
AUC_DIR = Path("results/auc")
RANKING_PATH = "results/gene_variance_ranking.csv"
TARGETS = ["genotype", "cluster"]
CLUSTER_N_GENES = 5000  
CLUSTER_K = 2
N_FOLDS = 5
METHOD_NAMES = {"logreg": "Logistic regression (L1)", "rf": "Random forest"}
TARGET_NAMES = {"genotype": "Genotype (WT vs Trem2 KO)", "cluster": "A3 cluster (k=2)"}


def get_cv():
    return StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def load_groups(samples):
    meta = load_metadata().loc[samples]
    return meta["refinebio_title"].str.split("-").str[0]


def load_xy(target, n_genes, cluster_method):
    X = samples_by_genes(n_genes)
    if target == "genotype":
        meta = load_metadata().loc[X.index]
        y = (meta["genotype"] == "trem2ko").astype(int)
    elif target == "cluster":
        labels = load_labels(cluster_method, CLUSTER_N_GENES, CLUSTER_K).loc[X.index]
        tissue = load_metadata().loc[X.index, "tissue"]
        cortex_cluster = labels[tissue == "cortex"].mode()[0]
        y = (labels != cortex_cluster).astype(int)
    else:
        raise ValueError(f"unknown target {target}")
    assert list(y.index) == list(X.index), "X and y samples are not aligned"
    return X, y.rename("y")


def gene_symbols():
    rk = pd.read_csv(RANKING_PATH)
    return rk.set_index("gene_id")["symbol"]


def save_predictions(method, target, n_genes, samples, y_true, y_pred, prob):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"sample": list(samples), "true": list(y_true), "pred": list(y_pred), "prob": list(prob)})
    df.to_csv(PRED_DIR / f"{method}_{target}_{n_genes}g.csv", index=False)


def save_signature(method, target, n_genes, gene_ids, scores):
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    sym = gene_symbols()
    df = pd.DataFrame({"gene_id": list(gene_ids), "score": list(scores)})
    df["symbol"] = df["gene_id"].map(sym)
    df = df.reindex(df["score"].abs().sort_values(ascending=False).index)
    df[["gene_id", "symbol", "score"]].to_csv(SIG_DIR / f"{method}_{target}_{n_genes}g.csv", index=False)
    return df


def save_auc(method, rows):
    AUC_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(AUC_DIR / f"{method}.csv", index=False)


def load_all_auc():
    return pd.concat([pd.read_csv(f) for f in sorted(AUC_DIR.glob("*.csv"))], ignore_index=True)


def parse_name(stem):
    method, target, g = stem.split("_")
    return method, target, int(g.rstrip("g"))