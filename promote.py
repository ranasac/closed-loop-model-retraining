"""
Decides whether to promote a candidate model over the active model.

Usage:
    python promote.py --candidate models/logistic_regression_20260523_2352.pkl

If all metrics are better or equal, updates registry.json moving the old active
to history, and setting the candidate to active.
"""
import argparse
import json
import os
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from modeling.models_api import BaseMlModel
from modeling.data_preprocessor import FraudDataPreprocessor
from config import REGISTRY_PATH, DATA_DIR
from dataloader.load_data import FraudDataLoader
from modeling.registry_models_api import ModelEntry


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def accuracy(model, X, y):
    return (model.predict(X) == y).mean()


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_scores = model.predict_proba(X)
    return BaseMlModel.evaluate(y, y_scores, target_precision=0.95, threshold=0.5)

def get_model_latency(model, X):
    import time
    start_time = time.time()
    model.predict(X)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    return latency_ms

def check_no_errors(model, X):
    try:
        model.predict(X)
        return True
    except Exception as e:
        print(f"Model prediction error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="Path to candidate .pkl")
    parser.add_argument("--data_filename", default="applications_v1.csv", help="Path to evaluation data CSV")
    parser.add_argument("--target-latency-in-ms", type=int, default=100, help="Maximum acceptable latency for model predictions")
    args = parser.parse_args()
    
    candidate_path_str = args.candidate
    candidate_path_obj = Path(candidate_path_str)
    data_filename = args.data_filename
    target_latency_in_ms = args.target_latency_in_ms

    # 1. Load active registry registry structure
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    # 2. Extract and verify matching candidate profile from history/manifest metrics
    candidate_meta = None
    candidate_index = None
    for idx, c in enumerate(registry.get("candidates", [])):
        if c["artifact_path"] == candidate_path_str:
            candidate_meta = ModelEntry.model_validate(c)
            candidate_index = idx
            break

    # Fail fast if candidate model metadata is not found in registry candidates list to prevent silent errors downstream. Candidate model must be registered in registry.json before promotion.
    if not candidate_meta:
        raise ValueError(f"Candidate model path {candidate_path_str} not found in registry candidates. Please ensure the candidate model is registered in registry.json before promotion.")

    active_path = registry["active"]["artifact_path"]
    active_model_meta = ModelEntry.model_validate(registry["active"])
    active_model = load_model(active_path)
    candidate_model = load_model(candidate_path_str)

    # Patches legacy model profiles to match modern scikit-learn namespaces safely
    for model_obj in [active_model, candidate_model]:
        classifier = model_obj.named_steps['classifier'] if hasattr(model_obj, 'named_steps') else model_obj
        if not hasattr(classifier, 'multi_class'):
            classifier.multi_class = 'deprecated'

    model_dict = {"active": (active_model, active_model_meta), "candidate": (candidate_model, candidate_meta)}

    for key, (model, meta) in model_dict.items():
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model {meta.version} does not have predict_proba method. Ensure it is a scikit-learn compatible classifier.")
        
        # load data
        data_loader = FraudDataLoader()
        input_features = data_loader.get_features_for_version(filename=meta.trained_on)
        print(f"Input features for {meta.trained_on}: {input_features}")
        df = data_loader.load_data(filename=meta.trained_on)

        # create preprocessor instance with input features for validation and preprocessing
        preprocessor = FraudDataPreprocessor(input_features=input_features)
        preprocessor.validate_input_features(df)    

        # get features and target
        X = df[input_features]
        y = df["label"]

        print("calculating evaluation metrics...")
        model_eval_report = evaluate_model(model, X, y)
        print("checking for prediction errors...")
        if not check_no_errors(model, X):
            raise ValueError(f"Model {meta.version} failed prediction error check.")
        
        model_latency = get_model_latency(model, X)
        model_eval_report["latency_ms"] = round(model_latency, 2)
        model_dict[key] = (model, meta, model_eval_report)
        print(f"{key.capitalize()} model evaluation report:\n{model_eval_report}")

    # 4. Strict Closed-Loop Operational Validation Barriers
    if ((model_dict['candidate'][2]['accuracy'] > model_dict['active'][2]['accuracy']) and 
        (model_dict['candidate'][2]['pr_auc'] >= model_dict['active'][2]['pr_auc']) and 
        (model_dict['candidate'][2]['recall_at_95precision'] >= model_dict['active'][2]['recall_at_95precision']) and
        (model_dict['candidate'][2]['latency_ms'] <= target_latency_in_ms)):
        
        print("PROMOTE")

        # Capture old active model layout 
        old_active = registry["active"].copy()
        
        # Initialize historical tracking block fallback safely if missing
        if "history" not in registry:
            registry["history"] = []
            
        # Push historical data profile into historical storage array
        registry["history"].append(old_active)

        # Update candidate metadata dictionary with actual live evaluated metrics 
        candidate_meta.accuracy = round(float(model_dict['candidate'][2]['accuracy']), 4)
        candidate_meta.pr_auc = round(float(model_dict['candidate'][2]['pr_auc']), 4)
        candidate_meta.recall_at_95precision = round(float(model_dict['candidate'][2]['recall_at_95precision']), 4)

        # Swap candidate to active seat
        registry["active"] = candidate_meta.model_dump()  # Convert back to dict for JSON serialization

        # Remove candidate entry cleanly out of selection list if it matches
        if candidate_index is not None:
            registry["candidates"].pop(candidate_index)

        # 5. Flush structured updates safely out to atomic storage format
        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
            
        print(f"Successfully migrated {candidate_meta.version} to active production reference inside registry.json.")
    else:
        print("KEEP ACTIVE")


if __name__ == "__main__":
    main()