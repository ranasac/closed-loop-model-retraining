import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(ROOT_DIR, "data"))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(ROOT_DIR, "models"))
REGISTRY_PATH = os.getenv("REGISTRY_PATH", os.path.join(ROOT_DIR, "registry.json"))


class VersionedDataDetails:
    """
    A mapping of data versions to their corresponding file paths.
    This can be used to track which dataset version was used for training each model.
    """
    def __init__(self, versioned_dataset_filename, features):
        self.versioned_dataset_filename = versioned_dataset_filename
        self.data_path = os.path.join(DATA_DIR, versioned_dataset_filename)
        self.features = features


APPLICATION_V1_DATA_DETAILS = VersionedDataDetails(
    versioned_dataset_filename="applications_v1.csv",
    features=[
    "application_completion_seconds",
    "hour_of_day",
    "email_domain_risk_score",
    "account_age_days",
    "num_applications_last_24h",
    "ip_location_mismatch_km",
    "is_vpn_or_proxy",
    "profile_trust_score"
    ]
)


APPLICATION_V2_DATA_DETAILS = VersionedDataDetails(
    versioned_dataset_filename="applications_v2.csv",
    features=[
    "application_completion_seconds",
    "hour_of_day",
    "email_domain_risk_score",
    "account_age_days",
    "num_applications_last_24h",
    "ip_location_mismatch_km",
    "is_vpn_or_proxy",
    "profile_trust_score"
    ]
)

versioned_data_mapping = {
    "applications_v1.csv": APPLICATION_V1_DATA_DETAILS,
    "applications_v2.csv": APPLICATION_V2_DATA_DETAILS
}








