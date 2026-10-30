# TrustScan - Fake Job Posting Detection

TrustScan is a machine learning system for detecting potentially fraudulent job postings. It analyzes job-posting text, metadata, and scam-related cues to estimate fraud risk.

## Features

- TF-IDF based text analysis
- Metadata and scam-cue features
- Logistic Regression, SVM, Random Forest and XGBoost models
- Stacked TrustScan model
- Group-aware evaluation
- Keyword-shift robustness testing
- Conformal-style triage: **Publish, Review, Hold**
- Prediction explanations

## Project Structure

```text
TrustScan/
├── preprocessing/
├── features/
├── models/
├── evaluation/
├── robustness/
├── conformal/
├── explainability/
├── tests/
├── dataset/
├── train.py
├── predict.py
├── demo.py
├── run_all.py
├── trustscan_model.joblib
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py --data fake_job_postings.csv --output trustscan_model.joblib
```

## Run Demo

```bash
python demo.py
```

## Batch Prediction

```bash
python predict.py new_postings.csv predictions.csv
```

## Run Full Evaluation

```bash
python run_all.py
```

## Output

TrustScan classifies postings into three risk zones:

- **Publish** — lower predicted risk
- **Review** — uncertain, manual verification recommended
- **Hold** — higher predicted risk

The predictions are risk estimates and should not be treated as proof that a job posting is genuine or fraudulent.

## Dataset

The project uses the **Employment Scam Aegean Dataset (EMSCAD)** / Fake Job Postings dataset.

## Research Focus

The project evaluates fraud detection using text, metadata, robustness testing, group-aware validation, and human-in-the-loop triage.
