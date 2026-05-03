
# Entity Level Labels and QA/QC
One thing we need to make sure is that, our labels are on the entity level. For example, if same entity (such as candidate) apply for different jobs, we provide same entity_id for all those events/ request_ids.

This automatically allows us to perform QA/QC of the labels, as we get labels from different sources to make sure we apply strategies like "max_voted_label" etc. or calculated alighnment_score and if alignment_score < 0.8, then either drop that label or use that label with **reduced sample weight** for our ML modeling.

*** Our ML models can only be as good as our labels ***

# Labels/ Feedback schema

```
request_ts,
request_id,
entity_id,
label (unclear, correct, incorrect),
label_provider_id
```

# Model Features schema
If an existing feature is updated, then create a new version of feature, for example (num_feat_1_v2) so that existing feature keeps on getting served to exising ML models in prod

```
request_ts,
request_id,
entity_id,
num_feat_1,
num_feat_2,
cat_feat_1,
cat_feat_2,
graph_embedding_1,
graph_embedding_2
```

# Training data
Join label_table with model_features table, to get the training data.
### Training data split
Create a test set which is out-of-time split using the latest data for testing. For the rest of data, perform a random GroupShuffleSplit based on entity_id, so that same entity_id cannot end up in both `train` and `val` set.


# Monitoring
Use Datadog (or Grafan etc.) for univariate monitoring, where the alerts thresholds are defined in a github repo using datadog (or similar software) SDK. Note: Important thing is we have a CI/CD framework for defining/ updating charts, metrics, thresholds and alerting settings. 
Have integration with Slack/ Pagerduty to raise alert whenever any metric exceeds threshold settings 

# Model/ data monitoring (Online/ Realtime)
1. Online/ realtime model monitoring
    - Flag_Rate monitoring (i.e. using predict_proba function)
        - Priority High
            - if rolling_avg_1min / rolling_avg_1day  > 1.8  ----> raise alert "too high flag_rate"
            - if rolling_avg_1min / rolling_avg_1day  < 0.2  ----> raise alert "too low flag_rate"
            - if flag_rate = 0. --> raise alert high priority "no ML flags raised"
        - Priority Medium
            - if rolling_avg_1min / rolling_avg_1day  > 1.5. ---> raise alert "flag rate 50% higher"
            - if rolling_avg_1min / rolling_avg_1day  < 0.5 ---> raise alert "flag rate 50% lower"
        - Priority Low
            - if rolling_avg_24hr / rolling_avg_72hr > 1.3. ----> raise alert "flag rates treding higher in last 1 day"
            - if rolling_avg_24hr / rolling_avg_72hr < 0.7. ----> raise alert "flag rates treding lower in last 1 day"
    
    - The flag_rate monitoring above doesn't tell us anything about drift in ML model scores below threshold in realtime. So, we need to monitor raw ML model scores as well.
        - Priority High
            - if median_model_score_1d / median_model_score_7d <0.5 or >1.5, --> raise alert "Median ML scores have shifted. Check input features distribution"

    2. Input features drift monitoring:
     We can either hand code it or use vendors like Whylabs, Anomalo that gives us out-of-the-box features anomaly monitoring. For numericals, it typically checks daily/ hourly means, for categoricals it checks daily/ hourly counts
        - Univariate monitoring
            - Perform `t-test` to check if hourly/ daily means for numerical are statistically different. If p-value < 0.05.  ---> raise alert "feature X may be drifing/ spiking"
        
        - Multivariate monitoring
            - Create 2nd order and 3rd order groups of features and check t-test results of those metrics. If p-value < 0.05.  ---> raise alert "feature X may be drifing/ spiking"

    3. Shap Values monitoring:
     If the relationship between model inputs and outputs have changed recently or if any feature calcuation logic has changed recently by us or by **vendors**, then we should have service to calculate shap_values (by taking a sample of data daily) and plot the shap values of top 50 features.
        - If avg_shap_value_last_1d / avg_shap_value_last_7d > 1.8 or < 0.3  ---> raise alert "Feature X impact to Model has potentially changed"


# Model/ data monitoring (Online/ Lagged for labeled feedback)
 Since feedback/ labels arrive after 1 or 2 weeks, we track the model metrics as labels arrive
    - Precision of flags
    - Recall of flags
    - Recall@ExpectedPrecision
    - Calibration curve ECE (expected calibration error)
If any of these metrics falls below 50% from value of previous 4 weeks numbers ---> raise alert "ML model performance degraded, Look at training new model"


# Model Retraining Trigger
For the situations below, a model retraining pipeline gets triggered either manually or by a click button on a UI (such as Streamlit app) or by publishing a Kafka event/ service queue etc. that automatically triggers a model training pipeline.
  # New data/features available
        1. New vendor data got ingested and featurized
        2. Existing feautres definitions got updated and feature versions incremented
        3. New cohort of users/ market became available for modeling
  # Model performance drops
        1. For metrics such as Recall@ExpectedPrecision, we see a drop larger than 30% for 2 consecutive days
  # Anomaly presence or significant drift
        1. If we see signicant spikes in model shap_values or model features t-test results (as shown above in monitoring section)
  # Set retraining cadence reached
        1. We usually set a default retraining cadence, such as if latest_model_trained_since > 2 months. ----> retrain a new model



# Multiple Models Framework
All we care about is correct flags and high recall at high precsion either coming from single model or multiple models. Instead of always trying replace the new ML model, we should have multiple models evaluating the input features and if any model flags, then flag that case. In this scenario, we get benefit of multiple models working in parallel, but we do need to manage and retire these models overtime using a model_manager strategy.


# Models Disagreement
When two models disagree on a given case, that may mean it is either a new trend, or it is an edge case, where both labels are likely given the histroical labels. In these cases, do the following
    - Shifting Trend: create a review queue to manually check these cases to confirm new trend. Otherwise, wait for some more new labeled data to arrive. Calculate the precision on the disagreed cases
    - Anomaly: check if anomaly detection metrics also fired, try to align the results
    - Edge cases: work on labels QA/QC and improve label quality


# Signals Disagreement
    - Check for latest signals code changes
    - Contact the data vendors if they changed something on their end
    - If signals look fine, then check the labels alignment score to make sure labels are correct
    **Remedy** - Create a heuristic rule that runs in parallel to ML model to override ML model decisions for particular signals combinations














# Label Alignment
