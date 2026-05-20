"""
Decides whether to promote a candidate model over the active model.

Usage:
    python promote.py --candidate models/model.pkl

If accuracy is higher than the active model, prints PROMOTE.
The actual promotion (updating registry.json, swapping the prod artifact)
is done by hand.
"""
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="Path to candidate .pkl")
    args = parser.parse_args()

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    active_path = registry["active"]["artifact_path"]
    active_model = load_model(active_path)
    candidate_model = load_model(args.candidate)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["label"]

    active_acc = accuracy(active_model, X, y)
    candidate_acc = accuracy(candidate_model, X, y)

    print(f"Active accuracy:    {active_acc:.4f}")
    print(f"Candidate accuracy: {candidate_acc:.4f}")

    if candidate_acc > active_acc:
        print("PROMOTE")
    else:
        print("KEEP ACTIVE")


if __name__ == "__main__":
    main()
