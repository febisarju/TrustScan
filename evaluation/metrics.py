"""Metrics, split generation, leakage auditing and benchmark evaluation."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score)

def metrics(y, s, thr):
    p, r, _ = precision_recall_curve(y, s)
    pred = (s >= thr).astype(int)
    return {"ROC-AUC": roc_auc_score(y, s), "PR-AUC": average_precision_score(y, s),
            "Recall@P90": float(r[p >= 0.90].max()) if (p >= 0.90).any() else 0.0,
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred)}

def get_splits(y, groups, protocol, n_splits=5, seed=42):
    if protocol == "random":
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))

def leakage_rate(text_hash, y, splits):
    a, f = [], []
    for tr, te in splits:
        seen = set(text_hash[tr]); hit = np.array([h in seen for h in text_hash[te]])
        a.append(hit.mean()); f.append(hit[y[te] == 1].mean())
    return float(np.mean(a)), float(np.mean(f))

def evaluate_protocol(data, y, groups, factories, protocol, seed=42, log=print):
    splits = get_splits(y, groups, protocol, seed=seed)
    rows, oof = [], {k: np.zeros(len(y)) for k in factories}
    for fi, (tr, te) in enumerate(splits):
        dtr, dte = data.take(tr), data.take(te)
        for name, fac in factories.items():
            m = fac().fit(dtr, y[tr], groups[tr]); s = m.score(dte); oof[name][te] = s
            rows.append({"protocol": protocol, "fold": fi, "model": name, **metrics(y[te], s, m.thr)})
        log(f"  [{protocol}] fold {fi+1}/{len(splits)} done")
    return pd.DataFrame(rows), oof