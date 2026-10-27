import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from a4_common import GENE_COUNTS, RANDOM_STATE, TARGETS, get_cv, load_groups, load_xy, save_auc, save_predictions, save_signature

METHOD = "rf"
CLUSTER_METHOD = "hclust" 
N_TREES = 500
TOP_N_SIGNATURE = 50


def make_model():
    return RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)


rows = []
for target in TARGETS:
    for n in GENE_COUNTS:
        X, y = load_xy(target, n, CLUSTER_METHOD)
        groups = load_groups(X.index)
        prob = cross_val_predict(make_model(), X, y, cv=get_cv(), groups=groups, method="predict_proba")[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y, prob)
        acc = (pred == y.values).mean()
        save_predictions(METHOD, target, n, X.index, y.values, pred, prob)
        model = make_model().fit(X, y)
        imp = model.feature_importances_
        keep = np.argsort(imp)[::-1][: min(TOP_N_SIGNATURE, n)]
        sig = save_signature(METHOD, target, n, X.columns[keep], imp[keep])
        rows.append({"method": METHOD, "target": target, "n_genes": n, "auc": auc,
                     "accuracy": acc, "signature_size": len(keep)})
        top = ", ".join(sig["symbol"].fillna(sig["gene_id"]).head(5))
        print(f"{target:8s} {n:>6,} genes  AUC={auc:.3f}  acc={acc:.3f}  top: {top}")

save_auc(METHOD, rows)
print("\nWrote results/predictions, results/signatures, results/auc/rf.csv")