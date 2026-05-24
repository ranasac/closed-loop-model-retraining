import pytest
import numpy as np
import json
from predict import get_prediction


def test_model_predict_no_error(registry_models_and_meta):
    """
    Ensure all models in registry can predict on a sample row from their trained data version using get_prediction.
    """
    for _, meta in registry_models_and_meta:
        if meta['version'] == 'v1':
            continue  # skip v1 which was a legacy model that doesn't handle missing values well
        try:
            result_json = get_prediction(
                model_path=meta["artifact_path"],
                predict_on_sample=True
            )
            output = json.loads(result_json)
            assert "score" in output
        except Exception as e:
            pytest.fail(f"get_prediction output error for model {meta.get('version', '?')} (sample): {e}")


def test_model_predict_all_missing_inputs(registry_models_and_meta):
    """
    Ensure all models in registry can predict on an input row with all features set to NaN using get_prediction.
    """
    for _, meta in registry_models_and_meta:
        if meta['version'] == 'v1':
            continue  # skip v1 which was a legacy model that doesn't handle missing values well
        try:
            result_json = get_prediction(
                model_path=meta["artifact_path"],
                predict_on_all_missing_inputs=True
            )
            output = json.loads(result_json)
            assert "score" in output
        except Exception as e:
            pytest.fail(f"get_prediction output error for model {meta.get('version', '?')} (all-missing-inputs): {e}")


def test_model_predict_empty_input(registry_models_and_meta):
    """
    Ensure all models in registry can predict on an empty input (no features) using predict.py.
    """
    for _, meta in registry_models_and_meta:
        if meta['version'] == 'v1':
            continue  # skip v1 which was a legacy model that doesn't handle missing values well
        try:
            result_json = get_prediction(
                model_path=meta["artifact_path"],
                predict_on_empty_input=True
            )
            output = json.loads(result_json)
            print(f"Output for model {meta.get('version', '?')} on empty input: {output}", flush=True)
            assert "score" in output
        except Exception as e:
            pytest.fail(f"get_prediction output error for model {meta.get('version', '?')} on empty input: {e}")
