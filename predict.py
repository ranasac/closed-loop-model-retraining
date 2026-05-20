"""
Scores a job application using the currently active model.

Usage:
    python predict.py  # scores a sample row

In prod this is wrapped in a Flask endpoint that reads registry.json
on startup and loads the active model. Predictions get logged to
data/predictions.csv (see the prediction log).
"""
import json
import pickle
from pathlib import Path

import pandas as pd

REGISTRY_PATH = Path(__file__).parent / "registry.json"
THRESHOLD = 0.5

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


def load_active_model():
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    with open(registry["active"]["artifact_path"], "rb") as f:
        return pickle.load(f), registry["active"]["version"]


def score(application: dict) -> dict:
    model, version = load_active_model()
    X = pd.DataFrame([application])[FEATURES]
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "model_version": version,
        "score": proba,
        "decision": "block" if proba >= THRESHOLD else "allow",
    }


if __name__ == "__main__":
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
    print(score(sample))
