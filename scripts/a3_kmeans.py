import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from a3_common import (
    GENE_COUNTS, K_RANGE, MAIN_N_GENES, RANDOM_STATE,
    describe_transition, load_metadata, samples_by_genes, save_chosen_k, save_labels,
)

METHOD = "kmeans"
CHOSEN_K = None  # None = pick the k with the best silhouette
N_INIT = 20


def fit_kmeans(X, k):
    return KMeans(n_clusters=k, init="k-means++", n_init=N_INIT, random_state=RANDOM_STATE).fit(X)


# Part A: k sweep on 5,000 genes
X_df = samples_by_genes(MAIN_N_GENES)
X = X_df.values
meta = load_metadata().loc[X_df.index]
print(f"Clustering {X.shape[0]} samples on {X.shape[1]} genes")

rows, labels_by_k = [], {}
for k in [1, *K_RANGE]:
    km = fit_kmeans(X, k)
    sil = silhouette_score(X, km.labels_) if k >= 2 else np.nan
    rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    if k >= 2:
        labels_by_k[k] = km.labels_
        save_labels(METHOD, MAIN_N_GENES, k, X_df.index, km.labels_)
    print(f"k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.3f}")

sweep = pd.DataFrame(rows)
sweep.to_csv("results/kmeans_k_sweep.csv", index=False)

chosen_k = CHOSEN_K or int(sweep.dropna().sort_values("silhouette", ascending=False).iloc[0]["k"])
save_chosen_k(METHOD, chosen_k)
print(f"Chosen k = {chosen_k}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(sweep["k"], sweep["inertia"], "o-")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[0].set_title("K-means elbow plot")
axes[1].plot(sweep["k"], sweep["silhouette"], "o-", color="tab:orange")
axes[1].axvline(chosen_k, ls="--", color="gray", label=f"chosen k = {chosen_k}")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Average silhouette width")
axes[1].set_title("K-means silhouette by k")
axes[1].legend()
for ax in axes:
    ax.set_xticks(sweep["k"])
fig.suptitle(f"K-means on top {MAIN_N_GENES:,} most variable genes (n={X.shape[0]} samples)")
fig.tight_layout()
fig.savefig("figures/kmeans_elbow_silhouette.png", dpi=150)
plt.close(fig)
print("Wrote figures/kmeans_elbow_silhouette.png")

# Part B: membership across k, and what each k lines up with
lines, transitions = [], []
lines.append("=== Consecutive k comparisons (5,000 genes) ===\n")
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

pd.DataFrame(transitions).to_csv("results/kmeans_k_transitions.csv", index=False)
with open("results/kmeans_crosstabs.txt", "w") as fh:
    fh.write("\n".join(lines))
print("Wrote results/kmeans_crosstabs.txt and results/kmeans_k_transitions.csv")

# Part C: gene count sweep at the chosen k
gene_rows = []
for n in GENE_COUNTS:
    Xn_df = samples_by_genes(n)
    km = fit_kmeans(Xn_df.values, chosen_k)
    save_labels(METHOD, n, chosen_k, Xn_df.index, km.labels_)
    sizes = np.bincount(km.labels_).tolist()
    sil = silhouette_score(Xn_df.values, km.labels_)
    gene_rows.append({"n_genes": n, "k": chosen_k, "silhouette": round(sil, 3), "cluster_sizes": sizes})
    print(f"{n:>6} genes  silhouette={sil:.3f}  sizes={sizes}")

pd.DataFrame(gene_rows).to_csv("results/kmeans_gene_sweep.csv", index=False)
print("Wrote results/kmeans_gene_sweep.csv")