import pandas as pd

wilcoxon = pd.read_csv("results/wilcoxon_enrichment_GO_Biological_Process_2021.csv")
gprofiler = pd.read_csv("results/enrichment_gprofiler2.csv")

wilcoxon = wilcoxon.rename(columns={
    "GO_ID": "native",
    "Term": "name",
    "P-value": "Wilcoxon_p",
    "FDR": "Wilcoxon_FDR",
    "Significant": "Wilcoxon_significant"
})

gprofiler = gprofiler.rename(columns={
    "p_value": "gProfiler_p",
    "significant": "gProfiler_significant"
})

wilcoxon = wilcoxon[["native", "name", "Wilcoxon_p", "Wilcoxon_FDR", "Wilcoxon_significant"]]
gprofiler = gprofiler[["native", "name", "gProfiler_p", "gProfiler_significant"]]

merged = pd.merge(wilcoxon, gprofiler, on=["native", "name"], how="outer")

merged["methods_included"] = (
    merged["Wilcoxon_p"].notna().astype(int) +
    merged["gProfiler_p"].notna().astype(int)
)

merged["methods_significant"] = (
    merged["Wilcoxon_significant"].fillna(False).astype(int) +
    merged["gProfiler_significant"].fillna(False).astype(int)
)

merged.to_csv("results/merged_enrichment_results.csv", index=False)