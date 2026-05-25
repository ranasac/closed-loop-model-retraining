# Video Demo Steps
These are the steps for the video demo.

## Start the API Locally
1. Using `make up`, start the local server. Make sure your `docker` is running on local machine.
2. See the home page at localhost:8000
3. Run a demo inference using an example payload:
    ```
    curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"data": {"application_completion_seconds": 45.0, "hour_of_day": 3, "email_domain_risk_score": 0.7, "account_age_days": 4, "num_applications_last_24h": 9, "ip_location_mismatch_km": 3200.0, "is_vpn_or_proxy": 1, "profile_trust_score": 0.2}}'
    ```
4. In production, this API would be called by something like a Kafka Consumer or it would listen to a Service Bus/ SQS/ RabbitMQ queue etc.



## Remote Deployment using Github CI/CD --> Show Github Actions Workflow
1. Upon push to `main` the workflow will trigger automatically and deploy the `active` model from the `registry.json` to google cloud run
2. By default it will only deploy on 10% traffic which we can increase later on.
3. For production monitoring, use Opentelemetry, Sentry, Datadog to log all metrics for dashboarding. 
    - Need to monitor metrics request count, error count, latency, custom metrics such as rolling_avg_ml_model_score_last_1hour, rolling_avg_flag_rate_last_1d etc.
4. We need to hook a database, in this case we can use a SQL database such as Postgress to save all prod data


## Code Walkthrough
Here is the code walkthrough

## Coding Principles
 1. Fail fast, fail during dev, avoid failures in prodution
 2. Test the models on edge cases such as missing inputs and missing features etc.
 3. `Decouple` the existing code, make it modular so it is easier to use, test and maintain. For example, create separate modules for data_loading, data_processing, model_building, model_training etc.
 4. Use `polymorphism` to allow training different types of model such as `logistic_regression or lightgbm` without require any code change, just by updating command line params
 4. Enforce data validation during writing to registry.json. Create standard schemas for registry_model
 5. Create mapping of versioned data to features in config.py. Using metadata of registry.json, automatically load appropriate feature columns using that mapping.
 6. Robust code to allow multiple versioned datasets for prediction. Seamless switch between models trained on different versions of data without worrying about code failures
 7. Make the api `stateless` so it can horizontally scaled using serverless architecture on the cloud
 8. Use docker to containerize the code to deploy on cloud
 9. Use `terraform` (if possible) for provisioning of cloud services
 10. Use github CI/CD for deployment
 11. For local development, use `uv` package manager for creating `venv`



### Show train_model.py
- talk about models_api.py. show off SOLID coding principles, using polymorphism to support multiple types of models training while keeping same api

- talk about data_loader.py

- talk about data_processor.py

- registry_modesl_api.py
    - standardize the registry model entries, perform automatic validation before writing to registry.json
    - if required data is not present, then fail writing to registry rather than it to fail at loading times

- **Steps**:
    - start with original `registry.json`

    - Train a new logistic regression model using `v1` dataset `python modeling/train_model.py --input-data-filename applications_v1.csv`
    - Train another logistic regression model using `v2` dataset `python modeling/train_model.py --input-data-filename applications_v2.csv`
        - show that we removed some features for `applications_v2.csv` and show how we handle it in config.py
    
    - Train another model using LightGBM model `python modeling/train_model.py --input-data-filename applications_v1.csv --model_type lightgbm`
    
    - Show the new models logged into `registry.json` automatically.
        - show `registry_models_api.py` to show standardized schema to write/ read from registry.json

    - Run promotion gate `python promote.py --candidate models/logistic_regression_20260524_1538.pkl`
        - If promoted, we should see the `registry.json` would get automatically updated
        - run promotion gate again on different versioned dataset to show cross versioned datasets trained model can be compared as well seamlessly
        - **Promotion Gate Logic**
             - Promote if candidate models accuracy, auc, recall@95Precision are higher than that of active model. 
            - Also, check if candidate model latency is under acceptable adhoc threshold of 100ms

    - Predict using `python predict.py --model models/logistic_regression_20260524_1538.pkl --predict-on-sample`

    - Predict on missing values dataset `python predict.py --model models/logistic_regression_20260524_1538.pkl --predict-on-all-missing-inputs`

    - Predict on missing values dataset `python predict.py --model models/logistic_regression_20260524_1538.pkl --predict-on-empty-input`

    - Show that the legacy model breaks down if missing inputs `python predict.py --model models/model.pkl --predict-on-all-missing-inputs` or `python predict.py --model models/model.pkl --predict-on-empty-input`


- **Tests**
    - The most important test we think is that, for all the models present in `registry.json`, make sure they are all healthy, and don't error out in normal as well as edge cases.
    - test all models for normal sample_data
    - test all modesl for edge cases where:
        - all inputs are NaN 
        - all features are missing i.e. empty payload
    - run tests using `make test`















