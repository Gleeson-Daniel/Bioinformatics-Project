import sys
import warnings
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
 
from a3_common import load_labels, load_metadata, load_subset
 
warnings.filterwarnings("ignore", message="Precision loss")
 
N_GENES_TESTED = 10000
PADJ_CUTOFF = 0.05
MIN_DIFF = 0.1
MAX_LIST = 500
MIN_TERM, MAX_TERM = 10, 500
MAPPING_PATH = "results/gene_id_mapping.csv"
RUN_ENRICH = "--no-enrich" not in sys.argv
 
 
def bh(p):
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out
 
 
def compare(vst, in_group, out_group, name):
    a = vst[in_group].to_numpy()
    b = vst[out_group].to_numpy()
    t, p = stats.ttest_ind(a, b, axis=1, equal_var=False)
    res = pd.DataFrame({
        "gene_id": vst.index,
        "mean_in": a.mean(axis=1),
        "mean_out": b.mean(axis=1),
        "t": t,
        "p": p,
    })
    res["diff"] = res["mean_in"] - res["mean_out"]
    res["padj"] = bh(res["p"].fillna(1))
    res["symbol"] = SYMBOLS.reindex(res["gene_id"]).values
    res = res.sort_values("p")
    res.to_csv(f"results/a3_cluster_de_{name}.csv", index=False)
 
    sig = res[(res["padj"] < PADJ_CUTOFF) & (res["diff"].abs() >= MIN_DIFF)]
    up = sig[sig["diff"] > 0].sort_values("t", ascending=False).head(MAX_LIST)
    down = sig[sig["diff"] < 0].sort_values("t").head(MAX_LIST)
    print(f"  {name}: {len(sig)} genes pass padj<{PADJ_CUTOFF} and |diff|>={MIN_DIFF} "
          f"({(sig['diff'] > 0).sum()} higher, {(sig['diff'] < 0).sum()} lower)")
    print(f"    top higher: {', '.join(up['symbol'].fillna(up['gene_id']).head(8))}")
    print(f"    top lower:  {', '.join(down['symbol'].fillna(down['gene_id']).head(8))}")
    return {"up": up["gene_id"].tolist(), "down": down["gene_id"].tolist()}
 
 
def enrich(genes, name):
    if len(genes) < 5:
        print(f"    {name}: only {len(genes)} genes, skipping enrichment")
        return None
    from gprofiler import GProfiler
    gp = GProfiler(return_dataframe=True)
    res = gp.profile(
        organism="mmusculus",
        query=genes,
        sources=["GO:BP"],
        background=BACKGROUND,
        domain_scope="custom_annotated",
        all_results=True,
    )
    res.to_csv(f"results/a3_enrichment_{name}.csv", index=False)
    sig = res[res["significant"] == True]
    specific = sig[(sig["term_size"] >= MIN_TERM) & (sig["term_size"] <= MAX_TERM)]
    print(f"    {name}: {len(genes)} genes, {len(res)} terms tested, {len(sig)} significant, "
          f"{len(specific)} specific ({MIN_TERM}-{MAX_TERM} genes)")
    for _, row in specific.head(5).iterrows():
        print(f"      {row['p_value']:.2e}  {row['name']}  ({row['intersection_size']}/{row['term_size']})")
    return specific
 
 
meta = load_metadata()
vst = load_subset(N_GENES_TESTED)
meta = meta.loc[vst.columns]
SYMBOLS = (pd.read_csv(MAPPING_PATH).drop_duplicates("ensembl_id")
           .set_index("ensembl_id")["symbol"])
BACKGROUND = vst.index.tolist()
print(f"VST: {vst.shape[0]} genes x {vst.shape[1]} samples")
 
lists = {}
 
k2 = load_labels("kmeans", 5000, 2).reindex(vst.columns)
print("\nK-means k=2 composition:")
print(pd.crosstab(k2, meta["tissue"]))
c1 = k2.index[k2 == 1].tolist()
c2 = k2.index[k2 == 2].tolist()
r = compare(vst, c1, c2, "k2_c1_vs_c2")
lists["k2_cluster1_higher"] = r["up"]
lists["k2_cluster2_higher"] = r["down"]
 
k8 = load_labels("kmeans", 5000, 8).reindex(vst.columns)
tab = pd.crosstab([k8, meta["tissue"]], meta["genotype"])
tab["n"] = tab.sum(axis=1)
tab["ko_frac"] = tab["trem2ko"] / tab["n"]
tissue_ko = (meta["genotype"] == "trem2ko").groupby(meta["tissue"]).mean()
tab["tissue_ko_frac"] = [tissue_ko[t] for _, t in tab.index]
tab["imbalance"] = (tab["ko_frac"] - tab["tissue_ko_frac"]).abs()
print("\nK-means k=8 composition:")
print(tab.round(2).to_string())
pick = tab[tab["n"] >= 20].sort_values("imbalance", ascending=False).index[0]
clus, tissue = pick
print(f"\nMost genotype-imbalanced k=8 cluster: {clus} ({tissue}), "
      f"KO fraction {tab.loc[pick, 'ko_frac']:.2f} vs {tab.loc[pick, 'tissue_ko_frac']:.2f} for all {tissue}")
same_tissue = meta.index[meta["tissue"] == tissue]
inside = [s for s in same_tissue if k8[s] == clus]
outside = [s for s in same_tissue if k8[s] != clus]
name = f"k8_c{clus}_vs_rest_{tissue}"
r = compare(vst, inside, outside, name)
lists[f"k8_c{clus}_higher"] = r["up"]
lists[f"k8_c{clus}_lower"] = r["down"]
 
if not RUN_ENRICH:
    print("\n--no-enrich given, stopping before g:Profiler")
    sys.exit(0)
 
print("\nEnrichment (g:Profiler GO:BP, custom background):")
tops = {}
for key, genes in lists.items():
    sig = enrich(genes, key)
    if sig is not None and len(sig):
        tops[key] = sig.head(8)
 
if tops:
    fig, axes = plt.subplots(len(tops), 1, figsize=(9, 2.6 * len(tops)), squeeze=False)
    for ax, (key, df) in zip(axes[:, 0], tops.items()):
        df = df.iloc[::-1]
        ax.barh(df["name"].str.slice(0, 55), -np.log10(df["p_value"]), color="#1f77b4")
        ax.set_title(key.replace("_", " "), fontsize=11, loc="left")
        ax.set_xlabel("-log10(adjusted p)")
        ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig("figures/a3_cluster_enrichment.png", dpi=150)
    print("\nWrote figures/a3_cluster_enrichment.png")
 