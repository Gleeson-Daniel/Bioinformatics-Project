import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

COUNTS_PATH = "data/SRP119064.tsv"
METADATA_PATH = "data/metadata_clean.tsv"

counts = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)
meta = pd.read_csv(METADATA_PATH, sep="\t")
meta = meta.set_index("refinebio_accession_code")
meta = meta.loc[counts.columns]

counts_filtered = counts[counts.sum(axis=1) >= 10]
print(f"Kept {counts_filtered.shape[0]} / {counts.shape[0]} genes after low-count filtering")

counts_t = counts_filtered.T.round().astype(int)

meta_for_dds = meta.copy()
for col in ["genotype", "tissue", "age"]:
    meta_for_dds[col] = meta_for_dds[col].astype(str)

meta_for_dds["genotype"] = pd.Categorical(
    meta_for_dds["genotype"], categories=["wt", "trem2ko"]
)

dds = DeseqDataSet(
    counts=counts_t,
    metadata=meta_for_dds,
    design="~tissue + age + genotype",
)
dds.deseq2()

stat_res = DeseqStats(dds, contrast=["genotype", "trem2ko", "wt"])
stat_res.summary()

stat_res.lfc_shrink(coeff="genotype[T.trem2ko]")

res = stat_res.results_df.copy()
res["threshold"] = res["padj"] < 0.05
res = res.sort_values("padj")

res.to_csv("results/full_de_results.csv")
res.head(50).to_csv("results/top50_de_genes.csv")

n_sig = int(res["threshold"].sum())
print(f"{len(res)} genes tested, {n_sig} significant at padj < 0.05")
print("Wrote results/full_de_results.csv and results/top50_de_genes.csv")

sig_genes = res[res["threshold"]].index.tolist()
with open("results/significant_genes.txt", "w") as fh:
    fh.write("\n".join(sig_genes))
print(f"Wrote results/significant_genes.txt ({len(sig_genes)} genes)")