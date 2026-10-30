"""Group-bootstrap test."""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

def group_bootstrap_diff(y, groups, s_a, s_b, n_boot=1000, seed=0, metric=average_precision_score):
    y, groups = np.asarray(y), np.asarray(groups)
    members = pd.Series(np.arange(len(y))).groupby(groups).apply(np.asarray).tolist()
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        if y[idx].min() == y[idx].max(): continue
        diffs.append(metric(y[idx], s_a[idx]) - metric(y[idx], s_b[idx]))
    d = np.array(diffs)
    return {"observed": metric(y, s_a) - metric(y, s_b), "ci95": tuple(np.percentile(d, [2.5, 97.5])),
            "p_better": float((d > 0).mean()), "n_boot": len(d)}

def main(a="TrustScan-Stack", b="LR-TFIDF"):
    meta = pickle.load(open("results/meta.pkl", "rb")); oof = pickle.load(open("results/benchmark.pkl", "rb"))["oof"]["group"]
    r = group_bootstrap_diff(meta["y"], meta["groups"], oof[a], oof[b])
    print(f"PR-AUC({a}) - PR-AUC({b}) = {r['observed']:+.3f}  95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
          f"P(better)={r['p_better']:.2f}  ({r['n_boot']} resamples)")
    print("If the CI contains 0, report the gain as not statistically distinguishable from the baseline.")

if __name__ == "__main__":
    main()