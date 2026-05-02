# Closed-Loop Model Retraining — Design Document

## Overview

This document describes a closed-loop system that connects a deployed binary classification model to its downstream feedback, enabling automated retraining, safe candidate evaluation, and reversible promotion. The design assumes a small team, delayed/incomplete labels, and high cost of bad predictions.

### Current State

| Component | Today | Target |
|---|---|---|
| Training | One-shot manual script | Scheduled, automated pipeline |
| Artifact upload | Manual copy | Versioned artifact store with lineage |
| Config promotion | Hand-edit config file | Gated promotion with automatic rollback |
| Prediction log | Exists, unused | Joined with feedback; drives retraining |
| Feedback table | Exists, unused | First-class input to training data |

---

## 1. System Design

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          CLOSED-LOOP PIPELINE                            │
│                                                                          │
│  ┌─────────┐    ┌──────────────┐    ┌──────────────┐                    │
│  │ Incoming │───▶│  Scoring API │───▶│  Downstream  │                    │
│  │ Records  │    │ (live model) │    │  Decision    │                    │
│  └─────────┘    └──────┬───────┘    └──────┬───────┘                    │
│                        │                    │                            │
│                        ▼                    │  (days–weeks later)        │
│               ┌────────────────┐            ▼                            │
│               │ Prediction Log │    ┌──────────────┐                    │
│               │  (append-only) │    │  Feedback     │                    │
│               │                │    │  Table        │                    │
│               │ • request_id   │    │ • request_id  │                    │
│               │ • features     │    │ • label       │                    │
│               │ • score        │    │ • feedback_ts │                    │
│               │ • model_version│    │ • source      │                    │
│               │ • scored_at    │    └──────┬───────┘                    │
│               └────────┬───────┘           │                            │
│                        │                   │                            │
│                        ▼                   ▼                            │
│               ┌────────────────────────────────────┐                    │
│               │        JOIN on request_id          │                    │
│               │  (scheduled daily materialization)  │                    │
│               └───────────────┬────────────────────┘                    │
│                               │                                          │
│                               ▼                                          │
│               ┌────────────────────────────┐                            │
│               │   Labeled Training Dataset  │                            │
│               │   (versioned snapshots)     │                            │
│               └───────────────┬────────────┘                            │
│                               │                                          │
│               ┌───────────────▼────────────┐                            │
│               │    Retraining Trigger?      │                            │
│               │  (weekly cron + label gate) │                            │
│               └───────────────┬────────────┘                            │
│                               │ yes                                      │
│                               ▼                                          │
│               ┌────────────────────────────┐                            │
│               │   Train Candidate Model    │                            │
│               │   (versioned artifact)     │                            │
│               └───────────────┬────────────┘                            │
│                               │                                          │
│                               ▼                                          │
│               ┌────────────────────────────┐                            │
│               │   Promotion Gate           │                            │
│               │  (offline eval + shadow)   │──── FAIL ──▶ alert, keep   │
│               └───────────────┬────────────┘             live model      │
│                               │ PASS                                     │
│                               ▼                                          │
│               ┌────────────────────────────┐                            │
│               │  Promote: swap model_version│                            │
│               │  in config (atomic pointer) │                            │
│               └───────────────┬────────────┘                            │
│                               │                                          │
│                               ▼                                          │
│               ┌────────────────────────────┐                            │
│               │   Post-deploy Monitoring   │──── ALERT ──▶ auto-rollback│
│               │   (drift, perf, latency)   │                            │
│               └────────────────────────────┘                            │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Feedback → Prediction Join

The prediction log and feedback table share a `request_id` foreign key. A **daily scheduled job** materializes the join into a labeled training view:

```sql
-- Materialized daily into `labeled_examples`
INSERT INTO labeled_examples
SELECT
    p.request_id,
    p.features,
    p.score,
    p.model_version,
    p.scored_at,
    f.label,           -- 'correct' | 'incorrect' | 'unclear'
    f.feedback_ts
FROM prediction_log p
INNER JOIN feedback f ON p.request_id = f.request_id
WHERE f.label IN ('correct', 'incorrect')   -- drop 'unclear'
  AND f.feedback_ts > (SELECT MAX(feedback_ts) FROM labeled_examples);
```

**Key decisions:**
- `unclear` labels are excluded from training but tracked for monitoring (high `unclear` rate may signal a labeling-process issue).
- The join is incremental (only new feedback rows) to keep the job fast.
- A `label_arrival_lag` metric (= `feedback_ts − scored_at`) is recorded so we can detect changes in feedback timing that would affect training freshness.

### 1.2 Retraining Trigger

Retraining is **not** triggered per-label. Instead, a **weekly cron** checks a gate:

```python
# Pseudocode: retraining trigger (runs weekly)
NEW_LABEL_THRESHOLD = 500          # minimum new labeled examples since last train
LABEL_STALENESS_CAP_DAYS = 30     # max age of newest label in training set

new_labels = count_labels_since(last_training_run.cutoff_ts)
newest_label_age = days_since(max(labeled_examples.feedback_ts))

should_retrain = (
    new_labels >= NEW_LABEL_THRESHOLD
    or newest_label_age >= LABEL_STALENESS_CAP_DAYS
)
```

- **Primary gate:** at least 500 new labeled rows since the last training run.
- **Staleness failsafe:** if no retraining has happened in 30 days (even if < 500 new labels), force a retrain to ensure the model doesn't silently age out.
- Both thresholds are config-driven and adjustable without code changes.

### 1.3 Training Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Snapshot     │───▶│  Train       │───▶│  Evaluate    │───▶│  Register    │
│  dataset      │    │  (same script│    │  (holdout +  │    │  artifact    │
│  (versioned)  │    │   as today)  │    │   metrics)   │    │  (versioned) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

- The existing training script is wrapped, not replaced. This minimizes adoption risk.
- Every run produces a **versioned dataset snapshot** (hash of row IDs + label counts) and a **versioned model artifact** (e.g., `model-v042-20260501.pkl`).
- Artifact metadata includes: dataset hash, hyperparameters, git SHA of training code, and all evaluation metrics.

### 1.4 Promotion and Rollback

**Promotion** is an atomic pointer swap: a single config value (`active_model_version`) is updated. The API loads models by version ID from an artifact store, so rollback is changing the pointer back.

```yaml
# model_config.yaml (version-controlled)
active_model_version: "v041"
previous_model_version: "v040"
rollback_enabled: true
```

- **Automated rollback:** If post-deploy monitors fire within the first 24 hours (see §3), the system reverts `active_model_version` to `previous_model_version` automatically.
- **Manual rollback:** Any team member can revert by changing one config value — no redeployment, no re-upload.
- Old artifacts are retained for at least 90 days so any prior version can be restored.

---

## 2. Promotion Gate

### Gate Pipeline

A candidate model must clear **all three stages** before it can be promoted:

```
              ┌─────────────────────┐
              │  Stage 1: Offline   │
              │  Holdout Eval       │──── FAIL ──▶ reject
              └─────────┬───────────┘
                        │ PASS
                        ▼
              ┌─────────────────────┐
              │  Stage 2: Stability │
              │  Checks             │──── FAIL ──▶ reject
              └─────────┬───────────┘
                        │ PASS
                        ▼
              ┌─────────────────────┐
              │  Stage 3: Shadow    │
              │  Scoring (48h)      │──── FAIL ──▶ reject
              └─────────┬───────────┘
                        │ PASS
                        ▼
              ┌─────────────────────┐
              │  PROMOTE            │
              └─────────────────────┘
```

### Stage 1 — Offline Holdout Evaluation

The candidate is evaluated on a **time-stratified holdout** (most recent 20% of labeled data, never seen during training). Metrics computed:

| Metric | Condition to pass | Rationale |
|---|---|---|
| AUC-ROC | ≥ live model AUC − 0.005 (non-inferiority) | Allows lateral moves that improve other properties |
| Precision @ operating threshold | ≥ live model precision | Downstream decision is costly to reverse; precision matters more than recall here |
| Recall @ operating threshold | ≥ live model recall − 0.03 | Small recall regression acceptable if precision holds |
| Brier score | ≤ live model Brier + 0.01 | Calibration matters because the fixed threshold assumes stable score distributions |
| Worst-slice accuracy | ≥ live model − 0.02 on every predefined segment | Prevents improvement in aggregate that hides regression on a subgroup |

**Statistical significance:** Each comparison uses a **paired bootstrap test** (n = 10,000 resamples) with α = 0.05. The candidate must show non-inferiority (the lower bound of the bootstrap CI for the metric difference must not cross the degradation threshold). We use bootstrap rather than asymptotic tests because holdout sizes may be modest (hundreds–low thousands).

```python
# Pseudocode: paired bootstrap non-inferiority test
import numpy as np

def is_non_inferior(metric_candidate, metric_live, threshold, n_bootstrap=10_000, alpha=0.05):
    """Returns True if candidate is non-inferior to live model."""
    diffs = []
    n = len(metric_candidate)
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        diff = metric_candidate[idx].mean() - metric_live[idx].mean()
        diffs.append(diff)
    lower_bound = np.percentile(diffs, 100 * alpha)
    return lower_bound >= threshold  # threshold is negative, e.g., -0.005
```

### Stage 2 — Stability Checks

Before shadow scoring, verify that the candidate behaves sanely:

- **Score distribution divergence:** KL divergence between candidate scores and live model scores on the same 10k-sample recent input set must be < 0.1. This catches models that are technically "accurate" on holdout but produce wildly different score distributions.
- **Threshold sensitivity:** Precision and recall must remain within ±0.02 of reported values when the threshold is perturbed by ±0.01. This catches models sitting on a cliff edge.
- **Latency:** P99 inference latency on benchmark inputs must be ≤ 1.2× live model P99. Prevents shipping a model that is accurate but too slow.

### Stage 3 — Shadow Scoring (48 hours)

The candidate runs **in shadow** alongside the live model for 48 hours. Both models score every request; only the live model's score is used for decisions. We compare:

- Agreement rate between candidate and live model decisions (same threshold). If < 90%, flag for human review before proceeding.
- Candidate's score distribution on live traffic vs. offline holdout. PSI > 0.1 suggests the holdout is not representative; halt promotion and investigate.

### When Signals Disagree

If the candidate passes offline eval but fails shadow (or vice versa):

| Scenario | Action |
|---|---|
| Offline pass, shadow agreement < 90% | **Do not auto-promote.** Page on-call. Likely distribution shift between holdout and live traffic. Investigate before re-running. |
| Offline pass, shadow PSI > 0.1 | **Do not auto-promote.** The holdout is stale. Rebuild holdout from more recent data and re-evaluate. |
| Offline marginal (passes on some metrics, fails on one by < 0.5× the threshold) | **Do not auto-promote.** Log for human review. May indicate noisy holdout; consider expanding holdout or waiting for more labels. |
| All stages pass but AUC is flat (< 0.001 improvement) | **Promote anyway.** The candidate is at least as good, and its training data is fresher. Staleness has real cost. |

---

## 3. Drift and Monitoring

### Monitored Signals

All signals are computed on a **rolling 24-hour window** and compared to a **30-day baseline**.

| Signal | Metric | Alert threshold | Action |
|---|---|---|---|
| **Feature drift** | PSI per feature | PSI > 0.2 on any of the top-10 importance features | Page on-call. Investigate upstream data pipeline. If confirmed real-world shift, trigger early retraining. |
| **Score distribution shift** | PSI on model output scores | PSI > 0.15 | Page on-call. If no feature drift, likely model degradation. If within 24h of promotion → auto-rollback. |
| **Prediction rate shift** | % positive predictions (at threshold) | > 2σ from 30-day mean | Page on-call. May indicate threshold miscalibration or population shift. |
| **Feedback-label rate** | Proportion of scored requests receiving eventual feedback | Drops below 50% of 30-day average | Alert (not page). Feedback pipeline may be broken; retraining quality at risk. |
| **Label distribution shift** | % positive labels in feedback | > 3σ from 30-day mean | Page on-call. Either real prevalence change or labeling process issue. |
| **Latency** | P99 scoring latency | > 2× baseline P99 or > absolute cap (e.g., 200ms) | Page on-call. May indicate infra issue or model complexity regression. |
| **Error rate** | HTTP 5xx rate from scoring API | > 1% of requests in 5-min window | Page on-call immediately. Likely infra/model-loading issue. Auto-rollback if within 24h of promotion. |

### Alert Escalation Flow

```
Signal exceeds threshold
        │
        ▼
┌───────────────────┐     yes     ┌──────────────────┐
│ Within 24h of a   │────────────▶│ AUTO-ROLLBACK     │
│ model promotion?  │             │ revert config to  │
└───────┬───────────┘             │ previous_version  │
        │ no                      └──────────────────┘
        ▼
┌───────────────────┐
│ PAGE ON-CALL      │
│ with dashboard    │
│ link + context    │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Human decision:   │
│ • rollback        │
│ • adjust threshold│
│ • trigger retrain │
│ • acknowledge     │
└───────────────────┘
```

### Concrete Example: Feature Drift Alert

> **Alert:** PSI on `transaction_amount_30d_avg` = 0.27 (threshold: 0.2). Rolling 24h vs 30-day baseline.
>
> **Dashboard shows:** Feature distribution shifted right — average values ~40% higher than baseline.
>
> **Runbook:**
> 1. Check upstream data source for schema changes or ETL bugs.
> 2. If data is correct (real-world shift), check model accuracy on recent feedback — is it degrading?
> 3. If accuracy is degrading, trigger early retrain (override weekly schedule).
> 4. If accuracy is stable, update baseline and suppress alert — the model is robust to this shift.

---

## 4. Explicit Non-Choices

### 4.1 Rejected: Real-Time Retraining (Online Learning)

**Considered:** Update the model incrementally as each feedback label arrives.

**Rejected because:**
- Labels arrive days to weeks late and are incomplete. An online learner would see a heavily biased, delayed stream.
- Debugging a model that has been updated thousands of times since an issue was introduced is dramatically harder than debugging a versioned batch-trained model.
- The team is small; online learning systems require sophisticated monitoring for concept drift, catastrophic forgetting, and label feedback loops. Batch retraining is boring — and boring is maintainable.

### 4.2 Rejected: A/B Testing for Promotion (Live Traffic Split)

**Considered:** Route a fraction of live traffic to the candidate model and compare real outcomes.

**Rejected because:**
- The downstream decision is costly to reverse. Exposing even 5% of traffic to a potentially bad model creates real business risk.
- Feedback delay (days–weeks) means an A/B test would need to run for weeks to get statistically significant outcome data. This is too slow for a weekly retrain cadence.
- Shadow scoring gives us the distribution-comparison signal of A/B testing without the risk, because the candidate scores but doesn't decide.

**Tradeoff acknowledged:** Shadow scoring cannot detect cases where the candidate's *decisions* would produce different feedback (counterfactual problem). We accept this and mitigate by requiring strong offline eval on labeled holdout.

### 4.3 Rejected: Automatic Threshold Tuning per Model

**Considered:** Re-optimize the decision threshold every time a new model is promoted.

**Rejected because:**
- Changing the threshold changes the decision boundary, which changes downstream behavior, which changes what feedback we receive. This creates a feedback loop that makes it very hard to reason about model quality vs. threshold quality.
- The fixed threshold is a known quantity the business has calibrated around. Changing it requires business sign-off, not automation.
- If calibration degrades (Brier score worsens), we surface that in the promotion gate and monitoring — but the response is to investigate, not auto-adjust.

### 4.4 Rejected: Complex Orchestrators (Kubeflow, Airflow DAGs)

**Considered:** A full ML platform with DAG orchestration for the retraining pipeline.

**Rejected because:**
- The team is small. The operational burden of maintaining Airflow/Kubeflow exceeds the benefit for a single-model, single-pipeline system.
- A cron job calling a Python script, with structured logging and a simple state table, covers our needs. Each step (join, trigger-check, train, evaluate, promote) is a standalone script that can be run and debugged independently.
- If the team or model count grows, this is the first thing to revisit.

### 4.5 Rejected: Multi-Armed Bandit for Model Selection

**Considered:** Use a bandit to dynamically allocate traffic between candidate and live model based on observed reward.

**Rejected because:**
- Reward (feedback label) arrives days–weeks later, making the bandit's explore/exploit loop impractically slow.
- Same risk exposure problem as A/B testing: the bandit must serve the candidate to real users.
- Adds algorithmic complexity without clear benefit given the feedback delay.

### 4.6 Rejected: Retraining on the Full Historical Dataset Every Cycle

**Considered:** Always train on all available labeled data from day one.

**Rejected because:**
- Older data may reflect a different distribution. Training on stale data dilutes the signal from recent examples.
- Training cost scales with dataset size; on a weekly cadence this becomes expensive.
- Instead, we use a **rolling training window** (e.g., last 12 months of labeled data) with recent data upweighted. The window size is a tunable parameter.

---

## Appendix: Key Configuration Parameters

All thresholds are centralized in a single config file to support easy tuning without code changes.

```yaml
retraining:
  schedule: "weekly"                  # cron cadence
  min_new_labels: 500                 # minimum new labels to trigger
  staleness_cap_days: 30              # force retrain if no run in N days
  training_window_months: 12          # rolling window for training data

promotion_gate:
  auc_non_inferiority_margin: 0.005
  precision_non_inferiority_margin: 0.0   # must match or beat
  recall_degradation_tolerance: 0.03
  brier_degradation_tolerance: 0.01
  worst_slice_degradation: 0.02
  bootstrap_resamples: 10000
  bootstrap_alpha: 0.05
  score_kl_divergence_max: 0.1
  threshold_sensitivity_delta: 0.01
  latency_multiplier_max: 1.2
  shadow_duration_hours: 48
  shadow_agreement_min: 0.90
  shadow_psi_max: 0.1

monitoring:
  feature_psi_alert: 0.2
  score_psi_alert: 0.15
  prediction_rate_sigma: 2.0
  feedback_rate_drop_pct: 50
  label_distribution_sigma: 3.0
  latency_multiplier_alert: 2.0
  latency_absolute_cap_ms: 200
  error_rate_pct: 1.0
  error_rate_window_min: 5
  auto_rollback_window_hours: 24

artifacts:
  retention_days: 90
```
