import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

COUNTS_PATH = "data/SRP119064.tsv"

counts = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)

log_counts = np.log2(counts + 1)
per_gene_median = log_counts.median(axis=1)

plt.figure(figsize=(7, 5))
sns.kdeplot(per_gene_median, fill=True)
plt.xlabel("Median log2(count + 1) expression per gene")
plt.ylabel("Density")
plt.title("Distribution of per-gene median expression")
plt.tight_layout()
plt.savefig("figures/density_median_expression.png", dpi=150)
plt.close()

print(f"Median of per-gene medians: {per_gene_median.median():.2f}")
print(f"IQR of per-gene medians: {per_gene_median.quantile(0.25):.2f} - {per_gene_median.quantile(0.75):.2f}")
print("Wrote figures/density_median_expression.png")