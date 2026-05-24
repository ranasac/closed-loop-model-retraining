import json
import os
import pandas as pd
import joblib
import pickle
from config import MODEL_DIR, REGISTRY_PATH
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
from modeling.registry_models_api import ModelEntry

import lightgbm as lgb
import numpy as np

from datetime import datetime
import re


class PandasCategoryEncoder(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    Custom transformer to cast specified columns to pandas 'category' type.
    Inherits from OneToOneFeatureMixin to natively support pipeline.set_output(transform="pandas").
    """
    def __init__(self, categorical_features=None):
        self.categorical_features = categorical_features or []

    def fit(self, X, y=None):
        # OneToOneFeatureMixin requires checking features during fit
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)
        return self

    def transform(self, X):
        # Work on a shallow copy to prevent modifying the original dataframe
        X_out = X.copy()
        for col in self.categorical_features:
            if col in X_out.columns:
                X_out[col] = X_out[col].astype('category')
        return X_out


class BaseMlModel:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.classifier = None
        self.pipeline = None

    def build_pipeline(self):
        raise NotImplementedError("Subclasses must implement build_pipeline()")
    
    @staticmethod     
    def evaluate(y_true, y_scores, target_precision=0.95, threshold=0.5):
        """Static method to evaluate model performance on test data."""
        y_scores = y_scores[:, 1] if y_scores.ndim > 1 else y_scores        

        y_pred = (y_scores >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recalls, precisions)
        
        valid_indices = np.where(precisions >= target_precision)[0]
        if len(valid_indices) > 0:
            recall_at_target_precision = np.max(recalls[valid_indices])
        else:
            recall_at_target_precision = 0.0
            
        return {
            "accuracy": accuracy,
            "pr_auc": pr_auc,
            f"recall_at_{int(target_precision * 100)}precision": recall_at_target_precision
        }
    

    def get_trained_classifier_metrics(self, X, y, target_precision=0.95):
        if self.pipeline is None:
            raise ValueError("Model must be trained or loaded before evaluation.")
        y_scores = self.predict_proba(X)
        return self.evaluate(y_true=y, y_scores=y_scores, target_precision=target_precision)
    
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        if self.pipeline is None:
            self.build_pipeline()
            
        fit_params = {}
        
        # If evaluation data is passed, let the subclass configure its early stopping details
        if X_val is not None and y_val is not None:
            fit_params = self._prepare_early_stopping(X_train, X_val, y_val)
            
        self.pipeline.fit(X_train, y_train, **fit_params)
        return self

    def _prepare_early_stopping(self, X_train, X_val, y_val):
        """
        Base method for early stopping setup. 
        Overridden by models that natively support it (like LightGBM).
        """
        return {}

    def predict(self, X):
        if self.pipeline is None:
            raise ValueError("Model must be trained or loaded before making predictions.")
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        if self.pipeline is None:
            raise ValueError("Model must be trained or loaded before making predictions.")
        return self.pipeline.predict_proba(X)

    def save_model(self, model_name):
        if self.pipeline is None:
            raise ValueError("No pipeline found to save. Train the model first.")
        
        os.makedirs(self.model_dir, exist_ok=True)
        filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
        
        # Open in write-binary mode ("wb")
        with open(filepath, "wb") as f:
            pickle.dump(self.pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"Model Pipeline successfully saved to {filepath}")


    def load_model(self, model_name):
        filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model found at {filepath}")
            
        # Open in read-binary mode ("rb")
        with open(filepath, "rb") as f:
            self.pipeline = pickle.load(f)
            
        # Keep internal reference to classifier sync'd with loaded pipeline
        self.classifier = self.pipeline.named_steps['classifier']
        print(f"Pipeline successfully loaded from {filepath}")

    
    def get_feature_names(self):
        """
        Extracts the final human-readable column names after the 
        data passes through the preprocessing pipeline steps.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be built or loaded first.")
            
        # Access the ColumnTransformer step ('selector' or 'preprocessor')
        preprocessor = self.pipeline.steps[0][1]
        
        # Scikit-learn transformers can return feature names natively
        try:
            return list(preprocessor.get_feature_names_out())
        except AttributeError:
            # Fallback if standard methods are missing
            return self.numerical_features + self.categorical_features
        
    
    def get_feature_importances_using_shap(self, X):
        """
        Generates global feature importance metrics using SHAP values.
        Automatically adapts to both Linear/Logistic models and Tree-based models.
        
        Returns:
            pd.DataFrame: Sorted feature names and their mean absolute SHAP values.
        """
        import pandas as pd
        import shap

        if self.pipeline is None:
            raise ValueError("Model must be trained or loaded before computing SHAP values.")

        # 1. Transform raw data through everything EXCEPT the final classifier step
        preprocessing_steps = self.pipeline[:-1]
        X_transformed = preprocessing_steps.transform(X)
        
        # Convert to a standard numpy array or keep dataframe if supported
        if hasattr(X_transformed, "values"):
            X_transformed_matrix = X_transformed.values
        else:
            X_transformed_matrix = X_transformed

        # 2. Extract the raw fitted classifier object
        raw_classifier = self.pipeline.named_steps['classifier']
        
        # 3. Dynamic Explainer Routing based on model type
        if isinstance(raw_classifier, lgb.LGBMClassifier):
            # TreeExplainer is lightning fast and handles trees directly
            explainer = shap.TreeExplainer(raw_classifier)
            shap_values = explainer.shap_values(X_transformed_matrix)
            
            # LightGBM outputs a list of arrays for binary/multiclass settings.
            # Index 1 corresponds to the positive class impact.
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                
        elif isinstance(raw_classifier, LogisticRegression):
            # LinearExplainer uses analytical shortcuts for linear/logistic models
            # Needs background training distribution data to calculate baseline drift
            explainer = shap.LinearExplainer(raw_classifier, X_transformed_matrix)
            shap_values = explainer.shap_values(X_transformed_matrix)
        else:
            # Universal model-agnostic explainer fallback
            explainer = shap.Explainer(raw_classifier, X_transformed_matrix)
            shap_values = explainer(X_transformed_matrix).values

        # 4. Collapse local row-level SHAP arrays into Global Feature Importances
        # We take the mean of the absolute values across all records
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # 5. Extract feature names and assemble summary report
        feature_names = self.get_feature_names()
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap
        })

        # Sort descending so the highest impact features sit at the top
        return importance_df.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    
    
    def register_model(self, model_name, trained_on, X_test, y_test, target_precision=0.95):
        """
        Registers a newly trained model as a candidate in the central registry using Pydantic,
        automatically enforcing type compliance, evaluating metrics, and incrementing versions.
        """
        from modeling.registry_models_api import RegistryModel, ModelEntry
        import re

        # 1. Load and parse existing registry data through Pydantic model validation
        if os.path.exists(REGISTRY_PATH):
            try:
                registry = RegistryModel.load_from_registry(REGISTRY_PATH)
            except (json.JSONDecodeError, Exception):
                # Fallback instance if the file is completely empty or corrupted
                registry = RegistryModel(active=None, candidates=[], history=[])
        else:
            registry = RegistryModel(active=None, candidates=[], history=[])

        # 2. Extract versions across Active, Candidates, and History to determine the next sequential version
        all_versions = []
        if registry.active:
            all_versions.append(registry.active.version)
        for candidate in registry.candidates:
            all_versions.append(candidate.version)
        for historical in registry.history:
            all_versions.append(historical.version)

        # Dig out integer version IDs (e.g., extracting 1 from "v1")
        version_numbers = []
        for v in all_versions:
            match = re.search(r'\d+', str(v))
            if match:
                version_numbers.append(int(match.group()))

        # Determine next integer index or default to 1 if the registry is brand new
        next_version_num = max(version_numbers) + 1 if version_numbers else 1
        next_version_str = f"v{next_version_num}"

        # 3. Calculate full structural evaluation metrics on your test set
        eval_metrics = self.get_trained_classifier_metrics(
            X_test, y_test, target_precision=target_precision
        )

        # 4. Resolve relative file path representations
        artifact_abs_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        artifact_relative_path = os.path.relpath(artifact_abs_path, start=os.getcwd())
        
        if os.path.isabs(str(trained_on)):
            trained_on_relative = os.path.relpath(str(trained_on), start=os.getcwd())
        else:
            trained_on_relative = str(trained_on)

        # 5. Package evaluation payload into a flat dictionary
        candidate_payload = {
            "version": next_version_str,
            "artifact_path": artifact_relative_path,
            "trained_on": trained_on_relative,
            "created_at": datetime.today().strftime('%Y-%m-%d'),
            "accuracy": round(float(eval_metrics.get("accuracy", 0.0)), 4),
            "pr_auc": round(float(eval_metrics.get("pr_auc", 0.0)), 4),
            "recall_at_95precision": round(float(eval_metrics.get(f"recall_at_{int(target_precision * 100)}precision", 0.0)), 4)
        }

        # Handle any extra metrics that might be present dynamically in eval_metrics
        for k, v in eval_metrics.items():
            if k not in ["accuracy", "pr_auc", f"recall_at_{int(target_precision * 100)}precision"]:
                candidate_payload[k] = round(float(v), 4)

        # 6. Instantiate via Pydantic ModelEntry to automatically trigger runtime type-checking
        new_candidate_model = ModelEntry(**candidate_payload)

        # Append candidate profile into the pool 
        registry.candidates.append(new_candidate_model)

        # 7. Atomic save using Pydantic's serialization machine
        with open(REGISTRY_PATH, "w") as f:
            # model_dump_json ensures native serialization of internal types
            f.write(json.dumps(registry.model_dump(), indent=2))

        print(f"Successfully registered model candidate as '{next_version_str}' in {REGISTRY_PATH}")
        print(f"Evaluated Test Metrics: {eval_metrics}")

    

class LogisticRegressionModel(BaseMlModel):
    def __init__(self, model_dir, max_iter=1000, categorical_features=None, numerical_features=None):
        super().__init__(model_dir)
        self.classifier = LogisticRegression(max_iter=max_iter)
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
    
    def build_pipeline(self):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', self.classifier)
        ])
        return self.pipeline


class LightGbmClassifierModel(BaseMlModel):
    def __init__(self, model_dir, categorical_features=None, numerical_features=None, **lgbm_params):
        super().__init__(model_dir)
        
        # Ensure we don't clobber parameters with defaults if user inputs custom ones
        params = {
            'n_estimators': 1000, # Increased max trees since early stopping will prune execution
            'learning_rate': 0.05,
            'objective': 'binary',
            'random_state': 42,
            'verbose': -1
        }
        params.update(lgbm_params)
            
        self.classifier = lgb.LGBMClassifier(**params)
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []

    def build_pipeline(self):
        all_features = self.numerical_features + self.categorical_features
        feature_selector = ColumnTransformer(
            transformers=[('keep_features', 'passthrough', all_features)],
            remainder='drop'
        )
        self.pipeline = Pipeline(steps=[
            ('selector', feature_selector),
            ('category_encoder', PandasCategoryEncoder(categorical_features=self.categorical_features)),
            ('classifier', self.classifier)
        ])
        self.pipeline.set_output(transform="pandas")
        return self.pipeline

    def _prepare_early_stopping(self, X_train, X_val, y_val):
        """
        Processes validation features through early transformers 
        and creates modern LightGBM metric callbacks.
        """
        # Slice pipeline to pull all steps preceding the 'classifier'
        preprocessing_steps = self.pipeline[:-1]
        
        # Fit-transform features on training layout first so state variables map securely
        preprocessing_steps.fit(X_train)
        
        # Process the clean validation matrices
        X_val_transformed = preprocessing_steps.transform(X_val)
        
        # Build modern optimization callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=True),
            lgb.log_evaluation(period=10)
        ]
        
        return {
            "classifier__eval_set": [(X_val_transformed, y_val)],
            "classifier__eval_metric": "auc",
            "classifier__callbacks": callbacks
        }