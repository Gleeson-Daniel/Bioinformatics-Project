import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from a4_common import GENE_COUNTS, METHOD_NAMES, SIG_DIR, TARGET_NAMES, TARGETS, load_all_auc, parse_name

auc = load_all_auc()
fig, ax = plt.subplots(figsize=(7.5, 4.8))
styles = {"genotype": "-", "cluster": "--"}
for (method, target), df in auc.groupby(["method", "target"]):
    df = df.sort_values("n_genes")
    ax.plot(df["n_genes"], df["auc"], styles[target], marker="o", lw=2,
            label=f"{METHOD_NAMES.get(method, method)}, {TARGET_NAMES[target]}")
ax.axhline(0.5, color="gray", ls=":", lw=1, label="Chance (AUC 0.5)")
ax.set_xscale("log")
ax.set_xticks(GENE_COUNTS)
ax.set_xticklabels([f"{n:,}" for n in GENE_COUNTS])
ax.set_ylim(0.35, 1.03)
ax.set_xlabel("Number of most variable genes used (log scale)")
ax.set_ylabel("Cross-validated AUC")
ax.set_title("Model performance by number of genes")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("figures/a4_auc_vs_genes.png", dpi=200)
print("Wrote figures/a4_auc_vs_genes.png")
sigs, symbols = {}, {}
for f in sorted(SIG_DIR.glob("*.csv")):
    key = parse_name(f.stem)
    df = pd.read_csv(f)
    sigs[key] = set(df["gene_id"])
    symbols.update(dict(zip(df["gene_id"], df["symbol"].fillna(df["gene_id"]))))
rows = []
methods = sorted({m for m, _, _ in sigs})
for method in methods:
    for target in TARGETS:
        for a, b in itertools.combinations(GENE_COUNTS, 2):
            A, B = sigs.get((method, target, a)), sigs.get((method, target, b))
            if A is None or B is None:
                continue
            shared = A & B
            rows.append({"method": method, "target": target, "genes_A": a, "genes_B": b,
                         "size_A": len(A), "size_B": len(B), "shared": len(shared),
                         "jaccard": round(len(shared) / len(A | B), 3),
                         "shared_genes": ", ".join(sorted(symbols[g] for g in shared))})
overlap = pd.DataFrame(rows)
overlap.to_csv("results/a4_signature_overlap.csv", index=False)
print(overlap.drop(columns="shared_genes").to_string(index=False))
rows = []
for target in TARGETS:
    for n in GENE_COUNTS:
        sets = {m: sigs.get((m, target, n)) for m in methods}
        if len(methods) < 2 or any(s is None for s in sets.values()):
            continue
        for m1, m2 in itertools.combinations(methods, 2):
            shared = sets[m1] & sets[m2]
            rows.append({"target": target, "n_genes": n, "method_A": m1, "method_B": m2,
                         "size_A": len(sets[m1]), "size_B": len(sets[m2]), "shared": len(shared),
                         "shared_genes": ", ".join(sorted(symbols[g] for g in shared))})
pd.DataFrame(rows).to_csv("results/a4_signature_method_overlap.csv", index=False)
appear = {}
for (m, t, n), genes in sigs.items():
    for g in genes:
        appear.setdefault(g, []).append(f"{m}_{t}_{n}g")
genes = pd.DataFrame({"gene_id": list(appear), "symbol": [symbols[g] for g in appear],
                      "n_signatures": [len(v) for v in appear.values()],
                      "signatures": ["; ".join(sorted(v)) for v in appear.values()]})
genes = genes.sort_values(["n_signatures", "symbol"], ascending=[False, True])
genes.to_csv("results/a4_signature_genes.csv", index=False)
print(f"\n{len(genes)} unique genes across {len(sigs)} signatures. Most frequent:")
print(genes.head(10)[["symbol", "n_signatures"]].to_string(index=False))