import gseapy as gp
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

INPUT_FILE = "results/full_de_results.csv"
ONTOLOGY = "GO_Biological_Process_2021"
ORGANISM = "Mouse"
MIN_SET_SIZE = 10     # skip terms with fewer measured genes than this
MAX_SET_SIZE = 500    # skip very broad terms
FDR_CUTOFF = 0.05

df = pd.read_csv(INPUT_FILE, index_col=0)
df = df.dropna(subset=["gene_symbol", "stat"])
df["gene_symbol"] = df["gene_symbol"].str.upper()   # library symbols are uppercase
df = df.drop_duplicates("gene_symbol")
stats = df.set_index("gene_symbol")["stat"]

print(f"{len(stats)} genes with a symbol and a test statistic")

gene_sets = gp.get_library(name=ONTOLOGY, organism=ORGANISM)
print(f"{len(gene_sets)} terms in {ONTOLOGY}")

terms = []
go_ids = []
set_sizes = []
pvalues = []
aucs = []

for term, genes in gene_sets.items():
    in_set = [g for g in set(genes) if g in stats.index]
    if len(in_set) < MIN_SET_SIZE or len(in_set) > MAX_SET_SIZE:
        continue
 
    inside = stats.loc[in_set]
    outside = stats.drop(index=in_set)
 
    u, p = mannwhitneyu(inside, outside, alternative="two-sided")
 
    terms.append(term)
    go_ids.append(term.split("(")[-1].replace(")", ""))   # ID is at the end of the name
    set_sizes.append(len(in_set))
    pvalues.append(p)
    aucs.append(u / (len(inside) * len(outside))) 

fdr = multipletests(pvalues, method="fdr_bh")[1]

results = pd.DataFrame({
    "GO_ID": go_ids,
    "Term": terms,
    "Set_size": set_sizes,
    "AUC": aucs,
    "Direction": ["up" if a > 0.5 else "down" for a in aucs],
    "P-value": pvalues,
    "FDR": fdr,

})

results["Significant"] = results["FDR"] < FDR_CUTOFF
results = results.sort_values("FDR")
 
out = f"results/wilcoxon_enrichment_{ONTOLOGY}.csv"
results.to_csv(out, index=False)
first_term = list(gene_sets)[0]
print(first_term, gene_sets[first_term][:5])
print(f"{len(results)} terms tested, {results['Significant'].sum()} significant at FDR < {FDR_CUTOFF}")
print(f"Saved to {out}")
print(results.head(10)[["Term", "Set_size", "Direction", "FDR"]].to_string(index=False))