import warnings

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from a4_common import GENE_COUNTS, TARGETS, get_cv, load_groups, load_xy, save_auc, save_predictions, save_signature

METHOD = "logreg"
CLUSTER_METHOD = "kmeans"  
C = 0.1  


def make_model():
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        lr = LogisticRegression(l1_ratio=1, solver="liblinear", C=C, max_iter=5000)
    else:
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000)
    return make_pipeline(StandardScaler(), lr)


warnings.filterwarnings("ignore", category=FutureWarning)
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
        coef = model[-1].coef_[0]
        keep = np.flatnonzero(coef)
        sig = save_signature(METHOD, target, n, X.columns[keep], coef[keep])
        rows.append({"method": METHOD, "target": target, "n_genes": n, "auc": auc,
                     "accuracy": acc, "signature_size": len(keep)})
        top = ", ".join(sig["symbol"].fillna(sig["gene_id"]).head(5))
        print(f"{target:8s} {n:>6,} genes  AUC={auc:.3f}  acc={acc:.3f}  signature={len(keep):3d}  top: {top}")

save_auc(METHOD, rows)
print("\nWrote results/predictions, results/signatures, results/auc/logreg.csv")