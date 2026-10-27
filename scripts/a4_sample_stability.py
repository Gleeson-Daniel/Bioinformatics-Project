import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
from a4_common import PRED_DIR, TARGETS, load_groups, parse_name


pred, prob, true = {t: {} for t in TARGETS}, {t: {} for t in TARGETS}, {}
for f in sorted(PRED_DIR.glob("*.csv")):
    method, target, n = parse_name(f.stem)
    df = pd.read_csv(f).set_index("sample")
    pred[target][f"{method}_{n}g"] = df["pred"]
    prob[target][f"{method}_{n}g"] = df["prob"]
    true[target] = df["true"]

order = lambda cols: sorted(cols, key=lambda c: (c.split("_")[0], int(c.split("_")[1].rstrip("g"))))
P = {t: pd.DataFrame(pred[t])[order(pred[t])] for t in TARGETS}
PR = {t: pd.DataFrame(prob[t])[order(prob[t])] for t in TARGETS}
for t in TARGETS:
    P[t].to_csv(f"results/a4_prediction_matrix_{t}.csv")
n_models = P["genotype"].shape[1]
print(f"{n_models} models per target: {list(P['genotype'].columns)}")

def stability(M):
    k = M.sum(axis=1)
    return np.maximum(k, M.shape[1] - k) / M.shape[1]

def confidence(M):
    return (2 * (M - 0.5).abs()).mean(axis=1)

samples = P["genotype"].index
counts = pd.DataFrame(index=samples)
counts["animal"] = load_groups(samples)
counts["true_genotype"] = true["genotype"].map({0: "WT", 1: "KO"})
counts["true_cluster"] = true["cluster"].map({0: "Cluster 1 (cortex)", 1: "Cluster 2 (hippocampus)"})
counts["n_models_WT"] = (P["genotype"] == 0).sum(axis=1)
counts["n_models_KO"] = (P["genotype"] == 1).sum(axis=1)
counts["n_models_cluster1"] = (P["cluster"] == 0).sum(axis=1)
counts["n_models_cluster2"] = (P["cluster"] == 1).sum(axis=1)
counts["genotype_stability"] = stability(P["genotype"])
counts["cluster_stability"] = stability(P["cluster"])
counts["genotype_frac_correct"] = P["genotype"].eq(true["genotype"], axis=0).mean(axis=1)
counts["cluster_frac_correct"] = P["cluster"].eq(true["cluster"], axis=0).mean(axis=1)
counts["genotype_confidence"] = confidence(PR["genotype"])
counts["cluster_confidence"] = confidence(PR["cluster"])
counts.to_csv("results/a4_sample_counts.csv")

print("\nGenotype stability (share of models agreeing), number of samples:")
print(counts["genotype_stability"].round(2).value_counts().sort_index().to_string())
print("\nCluster stability, number of samples:")
print(counts["cluster_stability"].round(2).value_counts().sort_index().to_string())

rows = []

def add_spearman(name, a, b, models):
    if a.nunique() < 2 or b.nunique() < 2:
        rows.append({"test": name, "models": models, "statistic": np.nan, "p": np.nan,
                     "note": "undefined: one variable is constant (every model agreed on every sample)"})
        return
    r, p = spearmanr(a, b)
    rows.append({"test": name, "models": models, "statistic": r, "p": p, "note": "Spearman rho"})

methods = sorted({c.split("_")[0] for c in P["genotype"].columns})
add_spearman("Label stability: genotype vs cluster", counts["genotype_stability"], counts["cluster_stability"], "all")
add_spearman("Confidence: genotype vs cluster", counts["genotype_confidence"], counts["cluster_confidence"], "all")
for m in methods:
    cols = [c for c in P["genotype"].columns if c.startswith(m + "_")]
    add_spearman("Label stability: genotype vs cluster", stability(P["genotype"][cols]), stability(P["cluster"][cols]), m)
    add_spearman("Confidence: genotype vs cluster", confidence(PR["genotype"][cols]), confidence(PR["cluster"][cols]), m)


animal = counts.groupby("animal")[["genotype_confidence", "cluster_confidence"]].mean()
r, p = spearmanr(animal["genotype_confidence"], animal["cluster_confidence"])
rows.append({"test": "Confidence: genotype vs cluster (animal averages, n=64)", "models": "all",
             "statistic": r, "p": p, "note": "Spearman rho"})

c1 = counts.loc[true["cluster"] == 0, "genotype_stability"]
c2 = counts.loc[true["cluster"] == 1, "genotype_stability"]
u, p = mannwhitneyu(c1, c2)
rows.append({"test": "Genotype stability: cluster 1 vs cluster 2", "models": "all", "statistic": u, "p": p,
             "note": f"Mann-Whitney U (median {c1.median():.2f} vs {c2.median():.2f})"})

tests = pd.DataFrame(rows)
ok = tests["p"].notna()
tests.loc[ok, "p_adj_BH"] = multipletests(tests.loc[ok, "p"], method="fdr_bh")[1]
tests.to_csv("results/a4_stability_tests.csv", index=False)
print("\n" + tests.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 8))
cmap = ListedColormap(["#c0392b", "#27ae60"])
for ax, t, title in zip(axes, TARGETS, ["Predicting genotype (WT vs KO)", "Predicting A3 cluster"]):
    correct = P[t].eq(true[t], axis=0).astype(int)
    correct = correct.loc[correct.mean(axis=1).sort_values(kind="stable").index]
    ax.imshow(correct.values, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(correct.shape[1]))
    ax.set_xticklabels([c.replace("logreg", "LR").replace("rf", "RF") for c in correct.columns], rotation=60, ha="right")
    ax.set_yticks([])
    ax.set_xlabel("Model (method, number of genes)")
    ax.set_ylabel(f"Samples (n={correct.shape[0]}), sorted by share of models correct")
    ax.set_title(title)
fig.legend(handles=[Patch(color="#27ae60", label="Correct"), Patch(color="#c0392b", label="Wrong")],
           loc="upper right", frameon=False)
fig.suptitle("Sample-specific predictions across all models (cross-validated)")
fig.tight_layout()
fig.savefig("figures/a4_prediction_matrix.png", dpi=200, bbox_inches="tight")
print("\nWrote figures/a4_prediction_matrix.png and results/a4_*.csv")