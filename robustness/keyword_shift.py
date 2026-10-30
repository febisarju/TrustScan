"""Keyword-removal robustness experiment."""
import numpy as np
import pandas as pd
from evaluation.metrics import get_splits, metrics
from models.classifiers import TextLR

def keyword_shift(data, y, groups, factories, Ks=(0, 50, 200, 500), seed=42, log=print):
    splits = get_splits(y, groups, "group", seed=seed); rows = []
    for fi, (tr, te) in enumerate(splits):
        dtr, dte = data.take(tr), data.take(te)
        models = {n: f().fit(dtr, y[tr], groups[tr]) for n, f in factories.items()}
        pos, _ = TextLR().fit(dtr, y[tr]).top_terms(max(Ks))
        for K in Ks:
            words = sorted({w for t, _ in pos[:K] for w in t.split()})
            dm = dte.mask(words)
            for n, m in models.items():
                rows.append({"fold": fi, "K": K, "model": n, **metrics(y[te], m.score(dm), m.thr)})
        log(f"  [shift] fold {fi+1}/{len(splits)} done")
    return pd.DataFrame(rows)