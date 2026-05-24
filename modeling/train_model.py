
import os
import sys
import argparse

# --- Ensure project root is in sys.path for imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from config import MODEL_DIR, DATA_DIR
from modeling.models_api import LogisticRegressionModel, LightGbmClassifierModel
from dataloader.load_data import FraudDataLoader
from modeling.data_preprocessor import FraudDataPreprocessor

# create enum for model types
class ModelType:
    LOGISTIC_REGRESSION = "logistic_regression"
    LIGHTGBM = "lightgbm"

def main(input_data_filename,
          features_to_drop=None,
          variance_threshold=0.0,
          target_column="label",
          model_type=ModelType.LOGISTIC_REGRESSION,
          target_precision=0.95):
    # Load data
    data_loader = FraudDataLoader()
    df = data_loader.load_data(filename=input_data_filename)
    input_features = data_loader.get_features_for_version(filename=input_data_filename)
    print(f"Input features for {input_data_filename}: {input_features}")

    # Preprocess data if needed (e.g., handle missing values, encode categorical features)
    preprocessor = FraudDataPreprocessor(features_to_drop=features_to_drop, variance_threshold=variance_threshold, input_features=input_features)
    df = preprocessor.preprocess_inputs(df)
    X, y = preprocessor.split_features_and_target(df, target_column=target_column)
    class_imbalance = preprocessor.get_class_imbalance(y)
    print(f"Class imbalance: {class_imbalance}")

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_train_test(X, y)
    categorical_features, numerical_features = preprocessor.get_categorical_and_numerical_features(X_train[input_features])
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")

    # Train and save Logistic Regression model
    if model_type == ModelType.LOGISTIC_REGRESSION:
        model = LogisticRegressionModel(model_dir=MODEL_DIR, categorical_features=categorical_features, numerical_features=numerical_features)
    elif model_type == ModelType.LIGHTGBM:
        model = LightGbmClassifierModel(model_dir=MODEL_DIR, categorical_features=categorical_features, numerical_features=numerical_features)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'logistic_regression' or 'lightgbm'.")
    model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    train_split_metrics = model.get_trained_classifier_metrics(X=X_train, y=y_train)
    val_split_metrics = model.get_trained_classifier_metrics(X=X_val, y=y_val)
    test_split_metrics = model.get_trained_classifier_metrics(X=X_test, y=y_test)
    print(f"Train split metrics: {train_split_metrics}")
    print(f"Validation split metrics: {val_split_metrics}")
    print(f"Test split metrics: {test_split_metrics}")

    # Compute Shapley values for feature importance
    shap_values = model.get_feature_importances_using_shap(X=X_val)
    print(f"SHAP values:\n{shap_values}")

    model_name = f"{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"

    model.save_model(model_name=model_name)

    # Log model details to registry
    model.register_model(model_name=model_name, trained_on=os.path.join(DATA_DIR, input_data_filename), X_test=X_test, y_test=y_test, target_precision=target_precision)


if __name__ == "__main__":
    # create argparse for params
    
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--input-data-filename", type=str, default="applications_v1.csv", help="Input data filename")
    parser.add_argument("--features_to_drop", type=str, nargs='*', default=None, help="Features to drop")
    parser.add_argument("--variance_threshold", type=float, default=0.0, help="Variance threshold for feature selection")
    parser.add_argument("--target_column", type=str, default="label", help="Target column name")
    parser.add_argument("--model_type", type=str, default=ModelType.LOGISTIC_REGRESSION, help="Type of model to train (logistic_regression or lightgbm)")
    parser.add_argument("--target_precision", type=float, default=0.95, help="Target precision for model registration")
    args = parser.parse_args()

    main(input_data_filename=args.input_data_filename,
          features_to_drop=args.features_to_drop,
            variance_threshold=args.variance_threshold,
              target_column=args.target_column,
                model_type=args.model_type,
                  target_precision=args.target_precision)