"""Runs every experiment and caches results to results/*.pkl."""
import sys, time, pickle, os, re, hashlib
import joblib
from features.engineering import FeatureBuilder, make_groups
from models.classifiers import make_models
from evaluation.metrics import leakage_rate, get_splits, evaluate_protocol
from robustness.keyword_shift import keyword_shift
from conformal.triage import run_conformal
from preprocessing.data import load_dataset
from train import train_bundle

def main():
    os.makedirs("results", exist_ok=True)
    csv = sys.argv[1] if len(sys.argv) > 1 else "fake_job_postings.csv"
    t0 = time.time(); log = lambda m: print(f"{time.time()-t0:7.0f}s {m}", flush=True)
    df = load_dataset(csv); y = df.fraudulent.values
    fb = FeatureBuilder().fit(df); data = fb.transform(df); groups = make_groups(df)
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
    text_hash = (df.title.fillna("") + df.description.fillna("") + df.requirements.fillna("")).map(
        lambda s: hashlib.md5(norm(s).encode()).hexdigest()).values
    with open("results/meta.pkl", "wb") as f: pickle.dump({"y": y, "groups": groups, "text_hash": text_hash, "fb": fb}, f)
    F = make_models()

    # 1) leakage audit + benchmark under two evaluation protocols
    leak = {p: leakage_rate(text_hash, y, get_splits(y, groups, p)) for p in ("random", "group")}
    res, oofs = [], {}
    import pandas as pd
    for p in ("random", "group"):
        r, o = evaluate_protocol(data, y, groups, F, p, log=log); res.append(r); oofs[p] = o
    with open("results/benchmark.pkl", "wb") as f: pickle.dump({"leak": leak, "bench": pd.concat(res), "oof": oofs}, f)
    log("benchmark saved")

    # 2) keyword-shift robustness (scammers avoid known words)
    sub = {k: F[k] for k in ("LR-TFIDF", "XGB-Meta+Cues", "TrustScan-Stack")}
    shift = keyword_shift(data, y, groups, sub, Ks=(0, 100, 300), log=log)
    with open("results/shift.pkl", "wb") as f: pickle.dump(shift, f)
    log("shift saved")

    # 3) conformal human-in-the-loop triage
    conf, kept = run_conformal(data, y, groups, sub, reps=4, keep="TrustScan-Stack", log=log)
    with open("results/conformal.pkl", "wb") as f: pickle.dump({"conf": conf, "kept": kept}, f)
    log("conformal saved")

    # 4) deployable model (same code path as train.py)
    bundle = train_bundle(df, fb=fb, data=data, groups=groups)
    joblib.dump(bundle, "trustscan_model.joblib")
    log(f"model saved low={bundle['t_low']:.3f} high={bundle['t_high']:.3f}")
    log("ALL DONE")

if __name__ == "__main__":  
    main()