"""
Scores a job application using the currently active model.

Usage:
    python predict.py  # scores a sample row

In prod this is wrapped in a Flask endpoint that reads registry.json
on startup and loads the active model. Predictions get logged to
data/predictions.csv (see the prediction log).
"""

import argparse
import json
import pickle
from pathlib import Path
import pandas as pd
from config import REGISTRY_PATH, versioned_data_mapping
from dataloader.load_data import FraudDataLoader
from modeling.registry_models_api import ModelEntry

THRESHOLD = 0.5



def load_model_and_meta(model_path=None):
    """
    Loads the model and its metadata (version, trained_on, etc).
    If model_path is None, loads the active model from registry.
    Returns: (model, meta_dict)
    """
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    if model_path is None:
        meta = registry["active"]
        model_path = meta["artifact_path"]
    else:
        # Try to find meta in candidates/history/active
        meta = None
        for section in ["candidates", "history"]:
            for entry in registry.get(section, []):
                if entry["artifact_path"] == model_path:
                    meta = entry
                    break
        if not meta and registry["active"]["artifact_path"] == model_path:
            meta = registry["active"]
        if not meta:
            raise ValueError(f"Model path {model_path} not found in registry. Please provide a valid model path or ensure the model is registered.")
        
    # get the model_entry object which will handle type coercion and validation
    model_meta = ModelEntry.model_validate(meta)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model, model_meta



def score(application: dict, model, features, version, threshold=THRESHOLD) -> dict:
    import numpy as np
    # Ensure all required features are present, fill missing with NaN
    row = {feat: application.get(feat, np.nan) for feat in features}
    import sys
    missing_features = [feat for feat in features if feat not in application]
    if missing_features:
        print(f"Warning: missing features in input will be filled with NaN: {missing_features}", file=sys.stderr)
    X = pd.DataFrame([row])[features]
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "model_version": version,
        "score": proba,
        "decision": "block" if proba >= threshold else "allow",
    }


def create_sample_input():
    sample = {
        "application_completion_seconds": 45.0,
        "hour_of_day": 3,
        "email_domain_risk_score": 0.7,
        "account_age_days": 4,
        "num_applications_last_24h": 9,
        "ip_location_mismatch_km": 3200.0,
        "is_vpn_or_proxy": 1,
        "profile_trust_score": 0.2,
    }
    return sample


def create_all_missing_input(features):
    import numpy as np
    return {feat: np.nan for feat in features}


def create_empty_input():
    return {}



def get_prediction(
    model_path=None,
    data=None,
    row=0,
    predict_on_sample=False,
    predict_on_all_missing_inputs=False,
    predict_on_empty_input=False,
    threshold=THRESHOLD
):
    model, model_meta = load_model_and_meta(model_path)
    # Patches legacy model profiles to match modern scikit-learn namespaces safely
    for model_obj in [model]:
        classifier = model_obj.named_steps['classifier'] if hasattr(model_obj, 'named_steps') else model_obj
        if not hasattr(classifier, 'multi_class'):
            classifier.multi_class = 'deprecated'

    trained_on = model_meta.trained_on
    import sys
    print(f"Model trained on: {trained_on}", file=sys.stderr)
    data_file = data if data else trained_on
    # Remove any leading directories for mapping lookup
    data_key = Path(data_file).name
    trained_on_key = Path(trained_on).name
    if data_key != trained_on_key:
        print(f"Warning: Data file {data_key} does not match model's trained_on {trained_on_key}. Ensure the correct data file is being used for scoring.", file=sys.stderr)
    if trained_on_key not in versioned_data_mapping:
        raise ValueError(f"Data file {trained_on_key} not found in versioned_data_mapping.")
    features = versioned_data_mapping[trained_on_key].features

    # Load data using dataloader
    data_loader = FraudDataLoader()
    if predict_on_sample:
        application = create_sample_input()
    elif predict_on_all_missing_inputs:
        application = create_all_missing_input(features)
    elif predict_on_empty_input:
        application = create_empty_input()
    else:
        df = data_loader.load_data(filename=data_key)
        if row >= len(df):
            raise IndexError(f"Row {row} is out of bounds for data file with {len(df)} rows.")
        application = df.iloc[row][features].to_dict()
    result = score(application, model, features, model_meta.version, threshold=threshold)
    return json.dumps(result)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score an application using a model.")
    parser.add_argument("--model", type=str, default=None, help="Path to model .pkl file (default: use active model in registry)")
    parser.add_argument("--data", type=str, default=None, help="Data CSV to score (default: use model's trained_on field). Example: applications_v1.csv")
    parser.add_argument("--row", type=int, default=0, help="Row index to score from the data file (default: 0)")
    parser.add_argument("--predict-on-sample", action="store_true", help="Whether to predict on a hardcoded sample input instead of loading data from CSV", default=False)
    parser.add_argument("--predict-on-all-missing-inputs", action="store_true", help="Predict on input containing all NaN values instead of loading data from CSV", default=False)
    parser.add_argument("--predict-on-empty-input", action="store_true", help="Predict on input containing no features instead of loading data from CSV", default=False)
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Threshold for block/allow decision (default: 0.5)")
    args = parser.parse_args()
    result_json = get_prediction(
        model_path=args.model,
        data=args.data,
        row=args.row,
        predict_on_sample=args.predict_on_sample,
        predict_on_all_missing_inputs=args.predict_on_all_missing_inputs,
        predict_on_empty_input=args.predict_on_empty_input,
        threshold=args.threshold
    )
    print(result_json)
