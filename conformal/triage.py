"""Group-aware calibration cutoffs, splits and human-in-the-loop triage."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def conformal_low(calib_fraud_scores, alpha):
    c = np.sort(calib_fraud_scores); k = int(np.floor(alpha * (len(c) + 1)))
    return -np.inf if k < 1 else c[k - 1]

def campaign_conformal_low(scores, y, groups, alpha):
    scores, y, groups = np.asarray(scores), np.asarray(y), np.asarray(groups)
    fraud = y == 1
    if not fraud.any():
        return -np.inf
    maxima = pd.DataFrame({"score": scores[fraud], "group": groups[fraud]}).groupby("group")["score"].max().to_numpy()
    maxima = np.sort(maxima)
    k = int(np.floor(alpha * (len(maxima) + 1)))
    return -np.inf if k < 1 else float(maxima[min(k - 1, len(maxima) - 1)])

def high_cutoff(s, y, target=0.90, min_flag=10):
    o = np.argsort(-s); tp = np.cumsum(y[o]); prec = tp / (np.arange(len(o)) + 1)
    ok = np.where((prec >= target) & (np.arange(len(o)) + 1 >= min_flag))[0]
    return s[o][ok.max()] if len(ok) else np.inf

def triage_zone(s, t_low, t_high): return np.where(s >= t_high, 2, np.where(s >= t_low, 1, 0))

def three_way_split(y, groups, seed):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    f = list(cv.split(np.zeros(len(y)), y, groups))
    te = f[0][1]; rest = f[0][0]
    cv2 = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
    a, b = next(iter(cv2.split(np.zeros(len(rest)), y[rest], groups[rest])))
    return rest[a], rest[b], te  

def run_conformal(data, y, groups, factories, alphas=(0.05, 0.10, 0.20), reps=5, seed=100, keep=None, log=print):
    rows, kept = [], {}
    for r in range(reps):
        tr, ca, te = three_way_split(y, groups, seed + r)
        for name, fac in factories.items():
            m = fac().fit(data.take(tr), y[tr], groups[tr])
            sc, st = m.score(data.take(ca)), m.score(data.take(te))
            th = high_cutoff(sc, y[ca])
            for a in alphas:
                tl = campaign_conformal_low(sc, y[ca], groups[ca], a); th2 = max(th, tl)
                z = triage_zone(st, tl, th2); yt = y[te]
                flagged = z >= 1
                rows.append({"rep": r, "model": name, "alpha": a, "target_recall": 1 - a,
                             "recall": flagged[yt == 1].mean(), "review_share": flagged.mean(),
                             "hold_share": (z == 2).mean(),
                             "hold_precision": yt[z == 2].mean() if (z == 2).any() else np.nan,
                             "missed_fraud": int(((z == 0) & (yt == 1)).sum()), "n_fraud": int(yt.sum()),
                             "n_test": len(te)})
            if keep and r == 0 and name == keep:
                kept = {"model": m, "tr": tr, "ca": ca, "te": te, "calib_scores": sc, "calib_y": y[ca], "test_scores": st}
        log(f"  [conformal] rep {r+1}/{reps} done")
    return pd.DataFrame(rows), kept

def weighted_low(fraud_scores, fraud_groups, alpha):
    s = np.asarray(fraud_scores); g = np.asarray(fraud_groups)
    w = 1.0 / pd.Series(g).map(pd.Series(g).value_counts()).values
    o = np.argsort(s); cum = np.cumsum(w[o]) / w.sum()
    i = np.searchsorted(cum, alpha, side="right")
    return -np.inf if i == 0 else s[o][i - 1] if i >= len(s) else s[o][i]

def group_recall(flagged, y, groups):
    m = y == 1
    return pd.Series(flagged[m]).groupby(groups[m]).mean().mean()