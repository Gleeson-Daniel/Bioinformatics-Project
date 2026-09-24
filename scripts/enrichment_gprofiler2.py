import pandas as pd
from gprofiler import GProfiler

with open("results/significant_genes_symbols.txt") as fh:
    sig_genes = [line.strip() for line in fh if line.strip()]

print(f"Running enrichment on {len(sig_genes)} significant genes")

gp = GProfiler(return_dataframe=True)
results = gp.profile(
    organism="mmusculus",
    query=sig_genes,
    sources=["GO:BP"],
    all_results=True,
)

results.to_csv("results/enrichment_gprofiler2.csv", index=False)

print(f"Tested {results.shape[0]} GO:BP terms")
if "significant" in results.columns:
    print(f"{int((results['significant'] == True).sum())} terms significant")
print("Wrote results/enrichment_gprofiler2.csv")