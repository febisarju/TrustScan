"""Export scores and readable reasons for a CSV."""
from predict import predict_dataframe, load_bundle
import pandas as pd

def explain_postings(csv_path, model_path="trustscan_model.joblib", output_path="explanations.csv"):
    result = predict_dataframe(pd.read_csv(csv_path), bundle=load_bundle(model_path))
    result.to_csv(output_path, index=False)
    return result