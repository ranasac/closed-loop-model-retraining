import sys
import os
# Ensure project root is in sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import json
import pickle
import os
import pytest
from pathlib import Path
from config import REGISTRY_PATH, versioned_data_mapping
from dataloader.load_data import FraudDataLoader

@pytest.fixture(scope="session")
def registry_models_and_meta():
    """
    Loads all models (active, candidates, history) from registry.json with their metadata.
    Returns a list of (model, meta_dict) tuples.
    """
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    all_entries = [registry["active"]]
    all_entries += registry.get("candidates", [])
    all_entries += registry.get("history", [])
    models = []
    for meta in all_entries:
        model_path = meta["artifact_path"]
        if not os.path.exists(model_path):
            continue  # skip missing artifacts
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        models.append((model, meta))
    return models

@pytest.fixture(scope="session")

def sample_data():
    """
    Loads a sample data row for each data version in versioned_data_mapping.
    Returns a dict: {data_key: sample_row_dict} and also with 'data/' prefix for compatibility.
    """
    loader = FraudDataLoader()
    samples = {}
    for data_key, details in versioned_data_mapping.items():
        df = loader.load_data(filename=data_key)
        if not df.empty:
            samples[data_key] = df.iloc[0][details.features].to_dict()
            samples[f"data/{data_key}"] = samples[data_key]
    return samples

@pytest.fixture(scope="session")

def edge_case_data():
    """
    Returns a dict of edge case input rows (all features set to None or NaN) for each data version.
    Also includes keys with 'data/' prefix for compatibility.
    """
    import numpy as np
    edge_cases = {}
    for data_key, details in versioned_data_mapping.items():
        edge_cases[data_key] = {feat: np.nan for feat in details.features}
        edge_cases[f"data/{data_key}"] = edge_cases[data_key]
    return edge_cases
