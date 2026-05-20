"""
Trains the job-applicant-fraud model.

Usage:
    python train.py

Outputs a pickle to models/model.pkl. To deploy, manually copy this file
to the prod artifact location and update registry.json by hand.
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_PATH = Path(__file__).parent / "data" / "applications_v1.csv"
MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"

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


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["label"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = (preds == y).mean()
    print(f"Accuracy: {accuracy:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
