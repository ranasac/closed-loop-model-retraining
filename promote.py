"""
Decides whether to promote a candidate model over the active model.

Usage:
    python promote.py --candidate models/logistic_regression_20260523_2352.pkl

If all metrics are better or equal, updates registry.json moving the old active
to history, and setting the candidate to active.
"""
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from modeling.models_api import BaseMlModel

REGISTRY_PATH = Path(__file__).parent / "registry.json"
DATA_PATH = Path(__file__).parent / "data" / "applications_v1.csv"

FEATURES = [
    "application_completion_seconds",
    "hour_of_day",
    "email_domain_risk_score",
    "account_age_days",
    "num_applications_last_24h",
    "ip_location_mismatch_km",
    "is_vpn_or_proxy",
    "profile_trust_score",
]


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def accuracy(model, X, y):
    return (model.predict(X) == y).mean()


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_scores = model.predict_proba(X)
    return BaseMlModel.evaluate(y, y_scores, target_precision=0.95, threshold=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="Path to candidate .pkl")
    args = parser.parse_args()
    
    candidate_path_str = args.candidate
    candidate_path_obj = Path(candidate_path_str)

    # 1. Load active registry registry structure
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    # 2. Extract and verify matching candidate profile from history/manifest metrics
    candidate_meta = None
    candidate_index = None
    for idx, c in enumerate(registry.get("candidates", [])):
        if c["artifact_path"] == candidate_path_str:
            candidate_meta = c
            candidate_index = idx
            break

    # Fallback structure just in case the filename passed wasn't declared in json ahead of time
    if not candidate_meta:
        candidate_meta = {
            "version": candidate_path_obj.stem,
            "artifact_path": candidate_path_str,
            "trained_on": str(DATA_PATH.relative_to(Path(__file__).parent)),
            "created_at": datetime.today().strftime('%Y-%m-%d')
        }

    active_path = registry["active"]["artifact_path"]
    active_model = load_model(active_path)
    candidate_model = load_model(candidate_path_str)

    # Patches legacy model profiles to match modern scikit-learn namespaces safely
    for model_obj in [active_model, candidate_model]:
        classifier = model_obj.named_steps['classifier'] if hasattr(model_obj, 'named_steps') else model_obj
        if not hasattr(classifier, 'multi_class'):
            classifier.multi_class = 'deprecated'

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["label"]

    # 3. Process complete mathematical diagnostics
    active_model_eval_report = evaluate_model(active_model, X, y)
    candidate_model_eval_report = evaluate_model(candidate_model, X, y)
    
    print(f"Active model evaluation report:\n{active_model_eval_report}")
    print(f"Candidate model evaluation report:\n{candidate_model_eval_report}")

    active_acc = accuracy(active_model, X, y)
    candidate_acc = accuracy(candidate_model, X, y)

    print(f"Active accuracy:    {active_acc:.4f}")
    print(f"Candidate accuracy: {candidate_acc:.4f}")

    # 4. Strict Closed-Loop Operational Validation Barriers
    if ((candidate_acc > active_acc) and 
        (candidate_model_eval_report["pr_auc"] >= active_model_eval_report["pr_auc"]) and 
        (candidate_model_eval_report["recall_at_95precision"] >= active_model_eval_report["recall_at_95precision"])):
        
        print("PROMOTE")

        # Capture old active model layout 
        old_active = registry["active"].copy()
        
        # Initialize historical tracking block fallback safely if missing
        if "history" not in registry:
            registry["history"] = []
            
        # Push historical data profile into historical storage array
        registry["history"].append(old_active)

        # Update candidate metadata dictionary with actual live evaluated metrics 
        candidate_meta["accuracy"] = round(float(candidate_acc), 4)
        candidate_meta["pr_auc"] = round(float(candidate_model_eval_report["pr_auc"]), 4)
        candidate_meta["recall_at_95precision"] = round(float(candidate_model_eval_report["recall_at_95precision"]), 4)

        # Swap candidate to active seat
        registry["active"] = candidate_meta

        # Remove candidate entry cleanly out of selection list if it matches
        if candidate_index is not None:
            registry["candidates"].pop(candidate_index)

        # 5. Flush structured updates safely out to atomic storage format
        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
            
        print(f"Successfully migrated {candidate_meta['version']} to active production reference inside registry.json.")
    else:
        print("KEEP ACTIVE")


if __name__ == "__main__":
    main()