"""Train and save a group-aware TrustScan model bundle."""
import argparse
import warnings
import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from preprocessing.data import load_dataset
from features.engineering import FeatureBuilder, make_groups
from models.classifiers import Stack
from conformal.triage import campaign_conformal_low, high_cutoff

def package_versions():
    import sklearn, xgboost, pandas
    return {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "pandas": pandas.__version__}

def train_bundle(df, alpha=0.10, seed=7, fb=None, data=None, groups=None):
    y = df["fraudulent"].to_numpy()
    groups = make_groups(df) if groups is None else groups
    fb = FeatureBuilder().fit(df) if fb is None else fb
    data = fb.transform(df) if data is None else data
    tr, ca = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups)))
    model = Stack().fit(data.take(tr), y[tr], groups[tr])
    scores = model.score(data.take(ca))
    low = campaign_conformal_low(scores, y[ca], groups[ca], alpha)
    if not np.isfinite(low):
        warnings.warn("Too few calibration fraud groups for this alpha: every posting will be sent to Review or Hold.")
    high = max(high_cutoff(scores, y[ca]), low)
    return {"fb": fb, "model": model, "t_low": low, "t_high": high, "alpha": alpha, "versions": package_versions()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="fake_job_postings.csv")
    parser.add_argument("--output", default="trustscan_model.joblib")
    args = parser.parse_args()
    bundle = train_bundle(load_dataset(args.data))
    joblib.dump(bundle, args.output)
    print(f"Saved model to {args.output}; low={bundle['t_low']:.3f}, high={bundle['t_high']:.3f}")

if __name__ == "__main__":
    main()