import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyComplexHeatmap import ClusterMapPlotter, HeatmapAnnotation, anno_simple

COUNTS_PATH = "data/SRP119064.tsv"
METADATA_PATH = "data/metadata_clean.tsv"

counts = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)
meta = pd.read_csv(METADATA_PATH, sep="\t")
meta = meta.set_index("refinebio_accession_code")
meta = meta.loc[counts.columns]

with open("results/significant_genes.txt") as fh:
    sig_genes = [line.strip() for line in fh if line.strip()]
sig_genes = [g for g in sig_genes if g in counts.index]

log_counts = np.log2(counts.loc[sig_genes] + 1)
z = log_counts.sub(log_counts.mean(axis=1), axis=0).div(log_counts.std(axis=1) + 1e-9, axis=0)

col_ha = HeatmapAnnotation(
    Genotype=anno_simple(meta["genotype"], cmap="Set2", legend=True),
    axis=1,
)

plt.figure(figsize=(9, 10))
cm = ClusterMapPlotter(
    data=z,
    top_annotation=col_ha,
    cmap="RdYlBu",
)
plt.savefig("figures/heatmap_sig_genes.png", bbox_inches="tight", dpi=150)
plt.close()

print(f"Wrote figures/heatmap_sig_genes.png ({len(sig_genes)} significant genes)")