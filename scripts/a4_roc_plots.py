import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from a4_common import GENE_COUNTS, METHOD_NAMES, PRED_DIR, TARGET_NAMES, TARGETS, load_all_auc, parse_name

preds = {parse_name(f.stem): pd.read_csv(f) for f in sorted(PRED_DIR.glob("*.csv"))}
methods = sorted({m for m, _, _ in preds})
colors = dict(zip(GENE_COUNTS, plt.cm.viridis([0.0, 0.25, 0.5, 0.72, 0.9])))

fig, axes = plt.subplots(len(methods), 2, figsize=(11, 4.8 * len(methods)), squeeze=False)
for i, method in enumerate(methods):
    for j, target in enumerate(TARGETS):
        ax = axes[i, j]
        for n in GENE_COUNTS:
            df = preds.get((method, target, n))
            if df is None:
                continue
            fpr, tpr, _ = roc_curve(df["true"], df["prob"])
            auc = roc_auc_score(df["true"], df["prob"])
            ax.plot(fpr, tpr, color=colors[n], lw=2, label=f"{n:,} genes (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="Chance (AUC 0.5)")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{METHOD_NAMES.get(method, method)}\npredicting {TARGET_NAMES[target]}")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_aspect("equal")
fig.suptitle("Cross-validated ROC curves (5 folds, grouped by biological sample)", y=1.0)
fig.tight_layout()
fig.savefig("figures/a4_roc_curves.png", dpi=200, bbox_inches="tight")
print("Wrote figures/a4_roc_curves.png")
auc = load_all_auc()
table = auc.pivot_table(index=["method", "target"], columns="n_genes", values="auc").round(3)
table.to_csv("results/a4_auc_table.csv")
print(table.to_string())