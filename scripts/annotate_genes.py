import pandas as pd
import mygene

COUNTS_PATH = "data/SRP119064.tsv"

counts = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)
ensembl_ids = counts.index.tolist()

mg = mygene.MyGeneInfo()

query_result = mg.querymany(
    ensembl_ids,
    scopes="ensembl.gene",
    fields="symbol",
    species="mouse",
    as_dataframe=True,
)

id_to_symbol = query_result["symbol"].dropna().to_dict()

n_mapped = len(id_to_symbol)
n_total = len(ensembl_ids)
print(f"Mapped {n_mapped} / {n_total} Ensembl IDs to gene symbols "
      f"({n_mapped / n_total:.1%})")

gene_symbol = counts.index.to_series().map(id_to_symbol)
mapping_df = pd.DataFrame({
    "ensembl_id": counts.index,
    "symbol": gene_symbol.values,
})
mapping_df.to_csv("results/gene_id_mapping.csv", index=False)
print("Wrote results/gene_id_mapping.csv")