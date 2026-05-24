import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from config import REGISTRY_PATH

class ModelEntry(BaseModel):
    # Enforces strict type conversion (e.g., handles string '1.0' into float 1.0 safely)
    version: str
    artifact_path: str
    trained_on: str
    created_at: str
    accuracy: float
    
    # Optional Fields with default values
    pr_auc: Optional[float] = None
    recall_at_95precision: Optional[float] = None

    # Allows the model to accept extra dynamic attributes gracefully
    model_config = ConfigDict(extra="allow")

    @property
    def extra_metrics(self) -> Dict[str, Any]:
        """
        Helper property to easily extract any dynamic extra metrics 
        that weren't explicitly defined as fields above.
        """
        # Get all fields present on the object
        current_data = self.model_dump()
        # Filter out the core defined schema properties
        defined_fields = {"version", "artifact_path", "trained_on", "created_at", "accuracy", "pr_auc", "recall_at_95precision"}
        return {k: v for k, v in current_data.items() if k not in defined_fields}


class RegistryModel(BaseModel):
    # If active is an empty dict, Pydantic converts it to None via validation hooks
    active: Optional[ModelEntry] = None
    candidates: List[ModelEntry] = Field(default_factory=list)
    history: List[ModelEntry] = Field(default_factory=list)

    @classmethod
    def load_from_registry(cls, path: str = REGISTRY_PATH) -> "RegistryModel":
        """
        Loads the registry.json file and parses it straight through 
        Pydantic's structural validation pipeline.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Registry file not found at: {path}")

        with open(path_obj, "r") as f:
            raw_data = json.load(f)

        # Handle edge case where "active" key exists but is empty/blank
        if "active" in raw_data and not raw_data["active"]:
            raw_data["active"] = None

        # model_validate passes your raw dictionary directly through type enforcement
        return cls.model_validate(raw_data)