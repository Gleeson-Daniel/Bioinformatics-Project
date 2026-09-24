import pandas as pd

df = pd.read_csv("results/merged_enrichment_results.csv")

top10 = df.sort_values(
    ["methods_significant", "methods_included"],
    ascending=False
).head(10)

top10.to_csv(
    "results/top10_enrichment_results.csv",
    index=False
)

print(top10.to_string(index=False))