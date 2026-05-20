# Closed-Loop Model Retraining — Starter Repo

A small job-applicant fraud-detection model deployed behind an API. It
flags applications that look like they came from bots, applicant farms,
or stolen/synthetic identities. This repo represents the status quo:
training is a one-shot script someone runs manually, the model artifact
is uploaded by hand, and `registry.json` is edited by hand to point
production at a new version.

See the assessment prompt for what we're asking you to build. This README
just covers what's already here and how to run it.

## Layout

```
.
├── data/
│   ├── applications_v1.csv    # original training data (8k rows)
│   ├── applications_v2.csv    # newer snapshot (12k rows)
│   ├── predictions.csv        # ~90 days of live model predictions
│   └── feedback.csv           # delayed labels from downstream reviewers
├── models/
│   └── model.pkl              # the currently active model
├── generate_data.py           # regenerates the datasets (don't need to run)
├── simulate_production.py     # regenerates the logs (don't need to run)
├── train.py                   # naive training script
├── promote.py                 # naive promotion gate
├── predict.py                 # represents the live scoring path
├── registry.json              # model registry (currently has v1 active)
└── requirements.txt
```

## Setup

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## Running what's here

Train a model (note: this overwrites `models/model.pkl`, which is what
the registry currently points at):

```bash
python train.py
```

Score a sample application with the active model:

```bash
python predict.py
```

Run the existing promotion gate against a candidate:

```bash
python promote.py --candidate path/to/candidate.pkl
```

## Data

`applications_v1.csv` and `applications_v2.csv` are two snapshots of
labeled job-application data. Features:

| column | description |
|---|---|
| `application_completion_seconds` | time spent filling out the application form |
| `hour_of_day` | 0–23, when the application was submitted |
| `email_domain_risk_score` | 0–1, from a third-party email reputation service (disposable/free-mail signal) |
| `account_age_days` | days since the candidate account was created |
| `num_applications_last_24h` | rolling count of applications submitted from the same identity |
| `ip_location_mismatch_km` | distance between the submission IP and the candidate's claimed location |
| `is_vpn_or_proxy` | 1 if the IP is a known VPN, proxy, or datacenter |
| `profile_trust_score` | 0–1, aggregated identity signal (LinkedIn presence, document verification, etc.) |
| `label` | 1 = fraud, 0 = legit |

`predictions.csv` is the live prediction log: every application the active
model has scored over the past ~90 days, with the input features and the
score it produced.

`feedback.csv` is what downstream reviewers (recruiters, hiring managers,
identity-verification ops) send back. Each row has a `prediction_id`
matching the prediction log, a `feedback_timestamp` (typically days to
weeks after the prediction), and a `verdict` of `fraud`, `legit`, or
`unclear`. Not every prediction gets feedback.

## Notes

- The repo is small on purpose. Don't feel obliged to use any of it as-is.
- Local files and JSON are fine for everything. No infrastructure required.
