"""Reusable TrustScan prediction functions and command-line interface.
The model score is a ranking/risk score; it is not necessarily a calibrated
probability. Human review remains required for consequential decisions.
"""
from __future__ import annotations
import argparse
import re
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from conformal.triage import triage_zone
from features.engineering import TEXT_FIELDS

CAT_FIELDS = ["department", "salary_range", "employment_type", "required_experience",
              "required_education", "industry", "function", "location"]
BOOL_FIELDS = ["telecommuting", "has_company_logo", "has_questions"]
REQUIRED_INPUT_FIELDS = TEXT_FIELDS + CAT_FIELDS + BOOL_FIELDS
_BLANK = {"", "nan", "none", "null"}
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
_PHONE = re.compile(r"(?<![\w$])(?:\+\d[\d\s().-]{7,}\d|\(?\d{3}\)?[-.\s]?\d{3}[-.]\d{4})(?!\w)")

def anonymise_raw(text: str) -> str:
    text = _EMAIL.sub(" #EMAIL_0# ", text)
    text = _URL.sub(" #URL_0# ", text)
    return _PHONE.sub(" #PHONE_0# ", text)

def _blank_to_nan(s: pd.Series) -> pd.Series:
    s = s.astype(object)
    blank = s.isna() | s.astype(str).str.strip().str.lower().isin(_BLANK)
    return s.mask(blank, np.nan)

def prepare_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    out = df.copy().reset_index(drop=True)
    for col in REQUIRED_INPUT_FIELDS:
        if col not in out.columns:
            out[col] = 0 if col in BOOL_FIELDS else np.nan
    for col in TEXT_FIELDS + CAT_FIELDS:
        out[col] = _blank_to_nan(out[col])
    for col in TEXT_FIELDS:
        out[col] = out[col].map(lambda x: x if pd.isna(x) else anonymise_raw(str(x)))
    for col in BOOL_FIELDS:
        if out[col].dtype == object:
            values = out[col].astype(str).str.strip().str.lower()
            out[col] = values.map({"true": 1, "yes": 1, "y": 1, "1": 1,
                                   "false": 0, "no": 0, "n": 0, "0": 0, "": 0})
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(0, 1).astype(int)
    return out

def _versions():
    import sklearn, xgboost
    return {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "pandas": pd.__version__}

def load_bundle(model_path="trustscan_model.joblib"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path.resolve()}. Train a model first with "
            "`python train.py --data fake_job_postings.csv --output trustscan_model.joblib`.")
    bundle = joblib.load(path)
    expected = {"fb", "model", "t_low", "t_high"}
    if not expected.issubset(bundle):
        raise ValueError("The saved file is not a compatible TrustScan model bundle. Retrain it with train.py.")
    saved = bundle.get("versions")
    if saved:
        now = _versions()
        diff = {k: (saved[k], now[k]) for k in saved if k in now and saved[k] != now[k]}
        if diff:
            warnings.warn(f"Model was trained with different package versions {diff}; retrain if results look odd.")
    return bundle

_LABELS = {"text_logit": "suspicious wording overall", "completeness": "incomplete posting",
           "caps_ratio": "many capital letters", "excl": "exclamation marks", "dollar": "dollar amounts",
           "digit_ratio": "many digits", "title_upper": "capitalised title"}

def describe_feature(name: str, value: float) -> str:
    if name in _LABELS: return _LABELS[name]
    if name == "has_company_logo": return "no company logo" if value == 0 else "company logo present"
    if name == "has_questions": return "no screening questions" if value == 0 else "screening questions present"
    if name == "telecommuting": return "remote / work-from-home"
    if name.startswith("miss_"): return f"missing {name[5:].replace('_', ' ')}"
    if name.startswith("cue_"): return f"{name[4:].replace('_', ' ')} wording"
    if name.startswith("n_"): return f"{name[2:].replace('tok', '')} present in text"
    if name.startswith("len_"): return f"unusual {name[4:].replace('_', ' ')} length"
    if "=" in name:
        field, val = name.split("=", 1)
        return f"{field.replace('_', ' ')}: {val}"
    return name.replace("_", " ")

def predict_dataframe(df: pd.DataFrame, bundle=None, model_path="trustscan_model.joblib") -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("Please provide at least one job posting.")
    bundle = bundle if bundle is not None else load_bundle(model_path)
    source = prepare_input_dataframe(df)
    data = bundle["fb"].transform(source)
    scores = np.asarray(bundle["model"].score(data), dtype=float)
    zones = np.array(["Publish", "Review", "Hold"])[triage_zone(scores, bundle["t_low"], bundle["t_high"])]
    reasons = [""] * len(source)
    flagged = np.where(zones != "Publish")[0]
    if len(flagged) and hasattr(bundle["model"], "explain"):
        try:
            for i, ex in zip(flagged, bundle["model"].explain(data.take(flagged))):
                feats = [describe_feature(str(f), v) for f, v, _ in ex.get("features", [])[:3]]
                terms = [str(t) for t in ex.get("terms", [])[:3]]
                reasons[i] = "; ".join(feats + terms)
        except Exception as exc:  
            warnings.warn(f"Explanations unavailable: {exc}")
            reasons = [""] * len(source)
    class_labels = {
        "Publish": "Likely legitimate (lower risk)",
        "Review": "Uncertain — manual review needed",
        "Hold": "Likely fraudulent (higher risk)",
    }
    return pd.DataFrame({
        "title": source["title"].fillna("").astype(str).to_numpy(),
        "prediction": [class_labels.get(zone, "Uncertain — manual review needed") for zone in zones],
        "fraud_score": np.round(scores, 4),
        "zone": zones,
        "why_flagged": reasons,
    })

def predict_csv(input_csv, output_csv="predictions.csv", model_path="trustscan_model.joblib"):
    result = predict_dataframe(pd.read_csv(input_csv), model_path=model_path)
    result.to_csv(output_csv, index=False)
    return result

def main():
    parser = argparse.ArgumentParser(description="Score job postings with TrustScan.")
    parser.add_argument("input_csv", help="CSV file containing one or more job postings")
    parser.add_argument("output_csv", nargs="?", default="predictions.csv", help="Output predictions CSV")
    parser.add_argument("--model", default="trustscan_model.joblib", help="Path to trained model bundle")
    args = parser.parse_args()
    result = predict_csv(args.input_csv, args.output_csv, args.model)
    print(result["zone"].value_counts().to_string())
    print(f"Saved: {args.output_csv}")

if __name__ == "__main__":
    main()