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
    X = pd.DataFrame([application])[features]
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "model_version": version,
        "score": proba,
        "decision": "block" if proba >= threshold else "allow",
    }



def main():
    parser = argparse.ArgumentParser(description="Score an application using a model.")
    parser.add_argument("--model", type=str, default=None, help="Path to model .pkl file (default: use active model in registry)")
    parser.add_argument("--data", type=str, default=None, help="Data CSV to score (default: use model's trained_on field)")
    parser.add_argument("--row", type=int, default=0, help="Row index to score from the data file (default: 0)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Threshold for block/allow decision (default: 0.5)")
    args = parser.parse_args()

    model, model_meta = load_model_and_meta(args.model)
    # Patches legacy model profiles to match modern scikit-learn namespaces safely
    for model_obj in [model]:
        classifier = model_obj.named_steps['classifier'] if hasattr(model_obj, 'named_steps') else model_obj
        if not hasattr(classifier, 'multi_class'):
            classifier.multi_class = 'deprecated'

    trained_on = model_meta.trained_on
    data_file = args.data if args.data else trained_on
    # Remove any leading directories for mapping lookup
    data_key = Path(data_file).name
    if data_key not in versioned_data_mapping:
        raise ValueError(f"Data file {data_key} not found in versioned_data_mapping.")
    features = versioned_data_mapping[data_key].features

    # Load data using dataloader
    data_loader = FraudDataLoader()
    df = data_loader.load_data(filename=data_key)
    if args.row >= len(df):
        raise IndexError(f"Row {args.row} is out of bounds for data file with {len(df)} rows.")
    application = df.iloc[args.row][features].to_dict()
    result = score(application, model, features, model_meta.version, threshold=args.threshold)
    print(result)

if __name__ == "__main__":
    main()
