import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

res = pd.read_csv("results/full_de_results.csv", index_col=0)
plot_df = res.dropna(subset=["padj", "log2FoldChange"]).copy()
plot_df["neg_log10_padj"] = -np.log10(plot_df["padj"].clip(lower=1e-300))

is_sig = plot_df["threshold"]

plt.figure(figsize=(7, 6))
plt.scatter(plot_df.loc[~is_sig, "log2FoldChange"], plot_df.loc[~is_sig, "neg_log10_padj"],
            s=8, color="grey", alpha=0.5, label="Not significant")
plt.scatter(plot_df.loc[is_sig, "log2FoldChange"], plot_df.loc[is_sig, "neg_log10_padj"],
            s=8, color="crimson", alpha=0.7, label="Significant (padj < 0.05)")
plt.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.8)
plt.xlabel("log2 fold change (Trem2-KO vs WT)")
plt.ylabel("-log10(adjusted p-value)")
plt.title("Volcano plot")
plt.legend()
plt.tight_layout()
plt.savefig("figures/volcano.png", dpi=150)
plt.close()

print("Wrote figures/volcano.png")