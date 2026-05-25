Here are the steps take in coding.

1. Environment creation:
    a. create `pyproject.toml` and use `uv` to create a `venv` to make sure the env is reproducible. If you don't have `uv` installed, then first install it on local machine.
    b. create a venv running command `uv venv` and then `source .venv/bin/activate`. Then run `uv sync`
    c. Now your local env should be fully setup.

2. The existing code is too coupled. So, first step is to de-couple the code. i.e. create separate files/ modules for data_loading, data_processing, model_training, model_eval, model_inference

3. Train, test split. Since I don't see time column in Input Data, so defaulting to random stratified split

4. The class imbalance exists. The positive class is only 4% frequency. This means, the baseline accuracy is 96%. Also, this means we should not use accuracy as our metric.

5. FastAPI
    curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"data": {"application_completion_seconds": 45.0, "hour_of_day": 3, "email_domain_risk_score": 0.7, "account_age_days": 4, "num_applications_last_24h": 9, "ip_location_mismatch_km": 3200.0, "is_vpn_or_proxy": 1, "profile_trust_score": 0.2}}'






