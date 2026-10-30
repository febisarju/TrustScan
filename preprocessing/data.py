"""Dataset loading and text normalization."""
from pathlib import Path
import pandas as pd
from features.engineering import normalise, TEXT_FIELDS

def load_dataset(path="fake_job_postings.csv"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    df = pd.read_csv(path)
    if "fraudulent" not in df.columns:
        raise ValueError("Dataset must contain a 'fraudulent' target column.")
    missing = [c for c in TEXT_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required text columns: {missing}")
    if df.empty:
        raise ValueError("Dataset contains no rows.")
    if df["fraudulent"].isna().any():
        raise ValueError("Target column 'fraudulent' contains missing values.")
    labels = set(pd.to_numeric(df["fraudulent"], errors="coerce").dropna().unique())
    if df["fraudulent"].isna().sum() or not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError("Target 'fraudulent' must contain both classes encoded as 0 and 1.")
    df["fraudulent"] = pd.to_numeric(df["fraudulent"], errors="raise").astype(int)
    return df

def combine_text(df):
    parts = [df[c].fillna("") for c in TEXT_FIELDS]
    return pd.Series([normalise(" . ".join(x)) for x in zip(*parts)], index=df.index)