import matplotlib
matplotlib.use("Agg")
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from a3_common import load_metadata, load_subset

genes = pd.read_csv("results/a4_signature_genes.csv")
meta = load_metadata()

H = load_subset(10000).loc[genes["gene_id"], meta.index]
H.index = genes["symbol"].values
print(f"Heatmap: {H.shape[0]} signature genes x {H.shape[1]} samples")

ann = pd.DataFrame({
    "Sample group (genotype, A1)": meta["genotype"].map({"wt": "Wild type", "trem2ko": "Trem2 KO"}),
    "Tissue": meta["tissue"].str.capitalize(),
})
palettes = {"Sample group (genotype, A1)": {"Wild type": "#1f77b4", "Trem2 KO": "#d62728"},
            "Tissue": {"Cortex": "#8c6d31", "Hippocampus": "#9edae5"}}
col_colors = pd.DataFrame({c: ann[c].map(palettes[c]) for c in ann})

g = sns.clustermap(
    H, z_score=0, cmap="vlag", center=0, vmin=-3, vmax=3,
    method="ward", metric="euclidean", col_colors=col_colors,
    xticklabels=False, yticklabels=True, figsize=(13, 16),
    dendrogram_ratio=(0.12, 0.08), cbar_kws={"label": "Gene z-score (VST)"},
)
g.ax_heatmap.tick_params(axis="y", labelsize=5)
g.ax_heatmap.set_xlabel(f"Samples (n={H.shape[1]})")
g.ax_heatmap.set_ylabel(f"Genes in any predictive signature (n={H.shape[0]})")
g.fig.suptitle("Predictive signature genes, all models and gene counts", y=1.01)
y = 1.0
for name, lut in palettes.items():
    leg = g.fig.legend(handles=[Patch(color=c, label=l) for l, c in lut.items()], title=name,
                       loc="upper left", bbox_to_anchor=(1.0, y), frameon=False)
    g.fig.add_artist(leg)
    y -= 0.08
g.fig.savefig("figures/a4_signature_heatmap.png", dpi=200, bbox_inches="tight")
print("Wrote figures/a4_signature_heatmap.png")