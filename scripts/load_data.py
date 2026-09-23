import pandas as pd

COUNTS_PATH = "data/SRP119064.tsv"
METADATA_PATH = "data/metadata_SRP119064.tsv"

counts = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)
metadata = pd.read_csv(METADATA_PATH, sep="\t")

print(f"Counts matrix: {counts.shape[0]} genes x {counts.shape[1]} samples")
print(f"Metadata: {metadata.shape[0]} samples")
print(f"Metadata columns: {list(metadata.columns)}")
