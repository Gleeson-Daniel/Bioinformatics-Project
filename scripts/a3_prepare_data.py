import pandas as pd
from pydeseq2.dds import DeseqDataSet

from a3_common import GENE_COUNTS, SUBSET_DIR, load_metadata

COUNTS_PATH = "data/SRP119064.tsv"
MAPPING_PATH = "results/gene_id_mapping.csv"
RANKING_PATH = "results/gene_variance_ranking.csv"

# Load and align
counts = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)
meta = load_metadata()

missing = set(counts.columns) - set(meta.index)
assert not missing, f"{len(missing)} count samples have no metadata row"
meta = meta.loc[counts.columns]
assert list(meta.index) == list(counts.columns)
print(f"Counts: {counts.shape[0]} genes x {counts.shape[1]} samples, metadata aligned")

# VST (same settings as pca.py)
counts_t = counts.T.round().astype(int)
meta_for_dds = meta.copy()
meta_for_dds["genotype"] = meta_for_dds["genotype"].astype(str)

dds = DeseqDataSet(counts=counts_t, metadata=meta_for_dds, design="~genotype")
dds.fit_size_factors()
dds.vst_fit(use_design=False)
vst_counts = dds.vst_transform()

# Back to genes x samples to match the refine.bio layout
vst = pd.DataFrame(vst_counts, index=counts_t.index, columns=counts_t.columns).T
assert not vst.isna().any().any(), "VST produced missing values"
print(f"VST matrix: {vst.shape[0]} genes x {vst.shape[1]} samples")

# Rank genes by variance across samples
gene_var = vst.var(axis=1).sort_values(ascending=False)
symbols = (
    pd.read_csv(MAPPING_PATH)
    .drop_duplicates("ensembl_id")
    .set_index("ensembl_id")["symbol"]
)
ranking = pd.DataFrame(
    {
        "gene_id": gene_var.index,
        "symbol": symbols.reindex(gene_var.index).values,
        "variance": gene_var.values,
        "rank": range(1, len(gene_var) + 1),
    }
)
ranking.to_csv(RANKING_PATH, index=False)
print(f"Wrote {RANKING_PATH}")

# Save the nested subsets
SUBSET_DIR.mkdir(parents=True, exist_ok=True)
for n in GENE_COUNTS:
    subset = vst.loc[gene_var.index[:n]]
    path = SUBSET_DIR / f"vst_top{n}.tsv.gz"
    subset.to_csv(path, sep="\t", compression="gzip", float_format="%.5f")
    print(f"Wrote {path}  shape={subset.shape}")

# Checks
for small, big in zip(GENE_COUNTS, GENE_COUNTS[1:]):
    assert list(gene_var.index[:small]) == list(gene_var.index[:big][:small])
print("Check passed: smaller gene sets are the top of larger ones")

print("\nTop 20 most variable genes:")
print(ranking.head(20)[["rank", "gene_id", "symbol", "variance"]].to_string(index=False))