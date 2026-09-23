import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from pydeseq2.dds import DeseqDataSet

COUNTS_PATH = "data/SRP119064.tsv"
METADATA_PATH = "data/metadata_clean.tsv"

counts = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)
meta = pd.read_csv(METADATA_PATH, sep="\t")
meta = meta.set_index("refinebio_accession_code")
meta = meta.loc[counts.columns]

counts_t = counts.T.round().astype(int)
meta_for_dds = meta.copy()
meta_for_dds["genotype"] = meta_for_dds["genotype"].astype(str)

dds = DeseqDataSet(counts=counts_t, metadata=meta_for_dds, design="~genotype")
dds.fit_size_factors()
dds.vst_fit(use_design=False)
vst_counts = dds.vst_transform()

vst_df = pd.DataFrame(vst_counts, index=counts_t.index, columns=counts_t.columns)
gene_var = vst_df.var(axis=0).sort_values(ascending=False)
top_genes = gene_var.head(2000).index
X = vst_df[top_genes].values

reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=15)
coords = reducer.fit_transform(X)

genotype = meta.loc[counts_t.index, "genotype"].values

plt.figure(figsize=(6.5, 5.5))
for g in sorted(set(genotype)):
    mask = genotype == g
    plt.scatter(coords[mask, 0], coords[mask, 1], label=g, alpha=0.75, s=35)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP")
plt.legend(title="Genotype")
plt.tight_layout()
plt.savefig("figures/umap.png", dpi=150)
plt.close()

print("Wrote figures/umap.png")