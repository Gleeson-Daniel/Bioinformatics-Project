import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from a3_common import (
    GENE_COUNTS, K_RANGE, MAIN_N_GENES,
    describe_transition, load_metadata, samples_by_genes, save_chosen_k, save_labels,
)

METHOD = "hclust"
LINKAGE = "ward"
CHOSEN_K = None  # None = pick the k with the best silhouette


def fit_hclust(X, k, linkage=LINKAGE):
    return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(X)


# Part A: k sweep on 5,000 genes
X_df = samples_by_genes(MAIN_N_GENES)
X = X_df.values
meta = load_metadata().loc[X_df.index]
print(f"Hierarchical ({LINKAGE}) on {X.shape[0]} samples x {X.shape[1]} genes")

rows, labels_by_k = [], {}
for k in K_RANGE:
    lab = fit_hclust(X, k).labels_
    sil = silhouette_score(X, lab)
    labels_by_k[k] = lab
    save_labels(METHOD, MAIN_N_GENES, k, X_df.index, lab)
    rows.append({"k": k, "silhouette": sil, "cluster_sizes": np.bincount(lab).tolist()})
    print(f"k={k:2d}  silhouette={sil:.3f}  sizes={np.bincount(lab).tolist()}")

sweep = pd.DataFrame(rows)
sweep.to_csv("results/hclust_k_sweep.csv", index=False)
chosen_k = CHOSEN_K or int(sweep.sort_values("silhouette", ascending=False).iloc[0]["k"])
save_chosen_k(METHOD, chosen_k)
print(f"Chosen k = {chosen_k}")

fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.plot(sweep["k"], sweep["silhouette"], "o-", color="tab:green")
ax.axvline(chosen_k, ls="--", color="gray", label=f"chosen k = {chosen_k}")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Average silhouette width")
ax.set_title(f"Hierarchical ({LINKAGE}) on top {MAIN_N_GENES:,} genes")
ax.set_xticks(sweep["k"])
ax.legend()
fig.tight_layout()
fig.savefig("figures/hclust_silhouette.png", dpi=150)
plt.close(fig)

# Part B: membership across k, and what each k lines up with
lines, transitions = ["=== Consecutive k comparisons (5,000 genes) ===\n"], []
ks = list(K_RANGE)
for k_prev, k_curr in zip(ks, ks[1:]):
    text, nested, ari = describe_transition(labels_by_k[k_prev], labels_by_k[k_curr], k_prev, k_curr)
    lines.append(text)
    transitions.append({"k_from": k_prev, "k_to": k_curr, "nested": nested, "ARI": round(ari, 3)})

lines.append("\n=== Each k vs sample groups ===\n")
for k in ks:
    for col in ["genotype", "tissue", "age"]:
        tab = pd.crosstab(pd.Series(labels_by_k[k] + 1, index=meta.index, name=f"cluster (k={k})"), meta[col])
        lines.append(f"{tab.to_string()}\n")

pd.DataFrame(transitions).to_csv("results/hclust_k_transitions.csv", index=False)
with open("results/hclust_crosstabs.txt", "w") as fh:
    fh.write("\n".join(lines))

# Part C: gene count sweep at the chosen k
gene_rows = []
for n in GENE_COUNTS:
    Xn_df = samples_by_genes(n)
    lab = fit_hclust(Xn_df.values, chosen_k).labels_
    save_labels(METHOD, n, chosen_k, Xn_df.index, lab)
    sil = silhouette_score(Xn_df.values, lab)
    sizes = np.bincount(lab).tolist()
    gene_rows.append({"n_genes": n, "k": chosen_k, "silhouette": round(sil, 3), "cluster_sizes": sizes})
    print(f"{n:>6} genes  silhouette={sil:.3f}  sizes={sizes}")
pd.DataFrame(gene_rows).to_csv("results/hclust_gene_sweep.csv", index=False)
print("Wrote results/hclust_*.csv, results/hclust_crosstabs.txt, figures/hclust_silhouette.png")