import pandas as pd

mapping = pd.read_csv("results/gene_id_mapping.csv").set_index("ensembl_id")["symbol"]

for name in ["full_de_results", "top50_de_genes"]:
    df = pd.read_csv(f"results/{name}.csv", index_col=0)
    if "gene_symbol" not in df.columns:
        df.insert(0, "gene_symbol", df.index.map(mapping))
        df.to_csv(f"results/{name}.csv")
        print(f"Updated results/{name}.csv with gene symbols")
    else:
        print(f"results/{name}.csv already has gene_symbol, skipping")

with open("results/significant_genes.txt") as fh:
    ensembl_ids = [line.strip() for line in fh if line.strip()]

symbols = []
for g in ensembl_ids:
    sym = mapping.get(g)
    symbols.append(g if pd.isna(sym) else sym)
n_mapped = sum(1 for g in ensembl_ids if pd.notna(mapping.get(g)))

with open("results/significant_genes_symbols.txt", "w") as fh:
    fh.write("\n".join(symbols))
print(f"Wrote results/significant_genes_symbols.txt ({n_mapped}/{len(ensembl_ids)} mapped)")