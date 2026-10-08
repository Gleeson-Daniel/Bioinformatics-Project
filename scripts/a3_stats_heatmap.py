import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from a3_common import CLUSTER_DIR, MAIN_N_GENES, load_chosen_k, load_labels, load_metadata, load_subset

GROUP_COL = "genotype"  # Assignment 1 sample groups (WT vs Trem2 KO)

# ---- gather every saved clustering result ----
meta = load_metadata()
results = {}
for f in sorted(CLUSTER_DIR.glob("*.csv")):
    method, g, k = f.stem.split("_")
    s = pd.read_csv(f).set_index("sample")["cluster"]
    results[f.stem] = s.reindex(meta.index)
res = pd.DataFrame(results)
print(f"{res.shape[1]} clustering results x {res.shape[0]} samples")


def chi(a, b):
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, np.nan, np.nan
    s, p, dof, _ = chi2_contingency(ct)
    return s, p, dof


# ---- chi-squared: each clustering vs Assignment 1 groups, and every pair of clusterings ----
rows = []
for c in res.columns:
    s, p, dof = chi(res[c], meta[GROUP_COL])
    rows.append({"comparison": "clusters vs genotype", "A": c, "B": GROUP_COL, "chi2": s, "dof": dof, "p": p})
for a, b in itertools.combinations(res.columns, 2):
    s, p, dof = chi(res[a], res[b])
    rows.append({"comparison": "clustering vs clustering", "A": a, "B": b, "chi2": s, "dof": dof, "p": p})
df = pd.DataFrame(rows)
ok = df["p"].notna()
df.loc[ok, "p_adj_BH"] = multipletests(df.loc[ok, "p"], method="fdr_bh")[1]  # one family: all tests together
df.to_csv("results/chisq_all.csv", index=False)
vs = df[df["comparison"] == "clusters vs genotype"].sort_values("p_adj_BH")
vs.to_csv("results/chisq_vs_genotype.csv", index=False)
print("\nClusterings vs genotype:")
print(vs[["A", "chi2", "dof", "p", "p_adj_BH"]].to_string(index=False))

# ---- same-k, same-method gene-count comparison (the 'effect of number of genes' table) ----
gene_pairs = df[
    (df["comparison"] == "clustering vs clustering")
    & df["A"].str.split("_").str[0].eq(df["B"].str.split("_").str[0])
    & df["A"].str.split("_").str[2].eq(df["B"].str.split("_").str[2])
    & ~df["A"].str.contains("5000g_k(?!2$)")
]
gene_pairs.to_csv("results/chisq_gene_count_pairs.csv", index=False)

# ---- heatmap: top 5,000 genes, n+1 annotation columns, both dendrograms ----
chosen = load_chosen_k()
H = load_subset(MAIN_N_GENES)[meta.index]
ann = pd.DataFrame({"Sample group (genotype)": meta[GROUP_COL]})
for method, label in [("kmeans", "K-means"), ("hclust", "Hierarchical (Ward)")]:
    k = chosen[method]
    ann[f"{label}, k={k}"] = load_labels(method, MAIN_N_GENES, k).reindex(meta.index).map(lambda x: f"Cluster {x}")

palettes = [sns.color_palette("Set1", 9), sns.color_palette("Dark2", 8), sns.color_palette("Set2", 8)]
col_colors, legends = pd.DataFrame(index=H.columns), []
for (name, ser), pal in zip(ann.items(), palettes):
    lut = dict(zip(sorted(ser.unique()), pal))
    col_colors[name] = ser.map(lut)
    legends.append((name, lut))

g = sns.clustermap(
    H, z_score=0, cmap="vlag", center=0, vmin=-3, vmax=3,
    method="ward", metric="euclidean", col_colors=col_colors,
    xticklabels=False, yticklabels=False, figsize=(13, 12),
    cbar_kws={"label": "Gene z-score (VST)"},
)
g.ax_heatmap.set_xlabel(f"Samples (n={H.shape[1]})")
g.ax_heatmap.set_ylabel(f"Top {MAIN_N_GENES:,} most variable genes")
y = 1.0
for name, lut in legends:
    leg = g.fig.legend(handles=[Patch(color=c, label=l) for l, c in lut.items()], title=name,
                       loc="upper left", bbox_to_anchor=(1.0, y), frameon=False)
    g.fig.add_artist(leg)
    y -= 0.14
g.fig.savefig("figures/heatmap_clusters_5000genes.png", dpi=170, bbox_inches="tight")
print("\nWrote figures/heatmap_clusters_5000genes.png and results/chisq_*.csv")