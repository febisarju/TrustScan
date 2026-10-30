"""TrustScan terminal demo: interactive single-job prediction or CSV batch scoring."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
from predict import load_bundle, predict_dataframe, REQUIRED_INPUT_FIELDS

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "trustscan_model.joblib"

def print_banner() -> None:
    print("\n" + "=" * 64)
    print("  TrustScan | Job Posting Fraud-Risk Screening")
    print("=" * 64)
    print("Research prototype.\n")

def prompt_for_posting() -> pd.DataFrame:
    fields = [
        ("title", "Job title"),
        ("company_profile", "Company profile"),
        ("description", "Job description"),
        ("requirements", "Requirements / qualifications"),
        ("benefits", "Benefits"),
        ("department", "Department"),
        ("salary_range", "Salary range"),
        ("employment_type", "Employment type"),
        ("required_experience", "Required experience"),
        ("required_education", "Required education"),
        ("industry", "Industry"),
        ("function", "Job function"),
        ("location", "Location"),
    ]
    row = {name: "" for name in REQUIRED_INPUT_FIELDS}
    print("Enter the job details. Press Enter to leave a field blank.")
    print("For the description, paste one line; multiline input is not supported.\n")
    for column, label in fields:
        row[column] = input(f"{label}: ").strip()

    print("\nOptional yes/no fields (enter y or n; blank defaults to n):")
    for column, label in [
        ("telecommuting", "Remote / telecommuting"),
        ("has_company_logo", "Company logo shown"),
        ("has_questions", "Screening questions included"),
    ]:
        value = input(f"{label}? [y/n]: ").strip().lower()
        row[column] = 1 if value in {"y", "yes", "1", "true"} else 0
    return pd.DataFrame([row])

def print_result(result: pd.DataFrame) -> None:
    for index, row in result.iterrows():
        print("\n" + "-" * 64)
        print(f"Job title       : {row.get('title') or '(not provided)'}")
        print(f"Prediction      : {row.get('prediction', 'Unavailable')}")
        print(f"Fraud risk score: {float(row['fraud_score']):.4f}")
        print(f"Decision        : {row.get('zone', 'Unknown')}")
        reason = str(row.get("why_flagged", "")).strip()
        print("Explanation     : " + (reason if reason else "No specific explanation signals returned."))
        print("-" * 64)
    print("\nInterpretation:")
    print("  Publish = lower model-estimated risk")
    print("  Review  = uncertain; manual verification recommended")
    print("  Hold    = higher model-estimated risk")
    print("\nThese are screening estimates.")
    print("Check the employer and posting independently before making decisions.\n")

def main() -> int:
    parser = argparse.ArgumentParser(description="Run TrustScan without Streamlit.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to a trained trustscan_model.joblib")
    parser.add_argument("--csv", help="Optional CSV containing one or more job postings")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path for batch mode")
    args = parser.parse_args()
    print_banner()
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()

    try:
        bundle = load_bundle(model_path)
    except FileNotFoundError:
        print("ERROR: The trained model file was not found.")
        print(f"Expected model: {model_path}")
        print("\nTrain the model first. Put the EMSCAD training CSV in the ML folder, then run:")
        print("  python train.py --data fake_job_postings.csv --output trustscan_model.joblib")
        print("\nReplace fake_job_postings.csv with the exact filename of your dataset.")
        return 2
    except Exception as exc:
        print(f"ERROR: Could not load the model: {exc}")
        print("If this model was created with different package versions, retrain it in this environment.")
        return 2

    try:
        if args.csv:
            input_path = Path(args.csv).expanduser()
            if not input_path.is_absolute():
                input_path = (Path.cwd() / input_path).resolve()
            if not input_path.exists():
                print(f"ERROR: Input CSV not found: {input_path}")
                return 2
            source = pd.read_csv(input_path)
            if source.empty:
                print("ERROR: The CSV contains no job postings.")
                return 2
            result = predict_dataframe(source, bundle=bundle)
            output_path = Path(args.output).expanduser()
            if not output_path.is_absolute():
                output_path = (Path.cwd() / output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_path, index=False)
            print(f"Processed {len(result)} posting(s) from: {input_path}")
            print_result(result)
            print(f"Predictions saved to: {output_path}")
        else:
            source = prompt_for_posting()
            result = predict_dataframe(source, bundle=bundle)
            print_result(result)
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"Could not process the input: {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    return 0

if __name__ == "__main__":
    sys.exit(main())