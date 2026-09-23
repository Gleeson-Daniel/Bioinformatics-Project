import pandas as pd

METADATA_PATH = "data/metadata_SRP119064.tsv"
OUT_PATH = "data/metadata_clean.tsv"

meta = pd.read_csv(METADATA_PATH, sep="\t")
parsed = meta["refinebio_subject"].str.split(",", expand=True)
meta["tissue"] = parsed[0].str.strip()
meta["genotype"] = parsed[1].str.strip()
meta["age"] = parsed[2].str.strip()

print(meta["genotype"].value_counts(dropna=False))
print(meta["tissue"].value_counts(dropna=False))
print(meta["age"].value_counts(dropna=False))

meta.to_csv(OUT_PATH, sep="\t", index=False)
print(f"Wrote {OUT_PATH}")