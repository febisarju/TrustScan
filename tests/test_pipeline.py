"""Fast smoke tests for preprocessing and feature construction."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from features.engineering import FeatureBuilder, cue_features, normalise
from preprocessing.data import load_dataset
from conformal.triage import campaign_conformal_low

def sample_frame():
    return pd.DataFrame({
        "title": ["Data analyst", "Urgent work from home"],
        "company_profile": ["Established employer", None],
        "description": ["Analyse reports for our team.", "Pay a registration fee now!"],
        "requirements": ["SQL skills", "No experience needed"],
        "benefits": ["Health insurance", ""],
        "department": ["Data", None],
        "salary_range": ["50000-60000", None],
        "employment_type": ["Full-time", "Contract"],
        "required_experience": ["Entry level", None],
        "required_education": ["Bachelor degree", None],
        "industry": ["Technology", None],
        "function": ["Analytics", None],
        "location": ["Chennai, India", "Remote"],
        "telecommuting": [0, 1],
        "has_company_logo": [1, 0],
        "has_questions": [1, 0],
        "fraudulent": [0, 1],
    })

class PipelineSmokeTests(unittest.TestCase):
    def test_placeholder_normalisation_and_cues(self):
        self.assertIn("urltok", normalise("Visit #URL_0#"))
        cues = cue_features(["Urgent! Pay a registration fee now"])
        self.assertGreater(cues.loc[0, "cue_urgency"], 0)
        self.assertGreater(cues.loc[0, "cue_payment"], 0)

    def test_feature_builder_preserves_schema(self):
        df = sample_frame()
        fb = FeatureBuilder().fit(df)
        train_data = fb.transform(df)
        new_data = fb.transform(df.iloc[[0]].copy())
        self.assertEqual(len(train_data), 2)
        self.assertEqual(len(new_data), 1)
        self.assertEqual(list(train_data.dense().columns), list(new_data.dense().columns))
        self.assertFalse(train_data.dense().isna().any().any())

    def test_campaign_cutoff_uses_group_maxima(self):
        cutoff = campaign_conformal_low(
            scores=[0.2, 0.9, 0.4, 0.8],
            y=[1, 1, 1, 1],
            groups=["A", "A", "B", "B"],
            alpha=0.5,)
        self.assertAlmostEqual(cutoff, 0.8)

    def test_dataset_loader_requires_both_binary_classes(self):
        df = sample_frame()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.csv"
            df.to_csv(path, index=False)
            loaded = load_dataset(path)
            self.assertEqual(set(loaded["fraudulent"]), {0, 1})
            df["fraudulent"] = 0
            df.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_dataset(path)

if __name__ == "__main__":
    unittest.main()