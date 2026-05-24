import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split



class BaseDataPreprocessor:
    def __init__(self):
        pass

    def preprocess_inputs(self, X):
        raise NotImplementedError("Subclasses must implement preprocess_inputs()")
    

class FraudDataPreprocessor(BaseDataPreprocessor):

    def __init__(self, features_to_drop=None, variance_threshold=0.0, input_features=None):
        super().__init__()
        self.features_to_drop = features_to_drop
        self.variance_threshold = variance_threshold
        self.categorical_features = None
        self.numerical_features = None
        self.input_features = input_features

    def validate_input_features(self, X):
        if self.input_features is not None:
            missing_features = set(self.input_features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Input features {missing_features} are missing from the dataset.") 
            

    def preprocess_inputs(self, X):
        # Implement any necessary preprocessing steps here (e.g., handling missing values, encoding categorical variables, feature engineering)
        # For example, you might want to fill missing values with the median for numerical features and the mode for categorical features, or create new features based on domain knowledge.
        self.validate_input_features(X)
        X = self.drop_features_adhoc_features(X)
        X = self.drop_features_containing_zero_variance(X)
        self.categorical_features, self.numerical_features = self.get_categorical_and_numerical_features(X[self.input_features])
        return X
    
    def get_features(self):
        return self.categorical_features, self.numerical_features
    
    def drop_features_adhoc_features(self, X):
        # drop features that are known to be uninformative or problematic based on domain knowledge
        # this is a simple way to improve model performance by removing noise from the data
        if self.features_to_drop is None or len(self.features_to_drop) == 0:
            return X
        return X.drop(columns=self.features_to_drop, errors='ignore')
        
    def drop_features_containing_zero_variance(self, X):
        # identify and drop features with zero variance
        # this is important because such features do not contribute to the model and can cause issues during training
        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(X)
        return X[X.columns[selector.get_support(indices=True)]]
    
    def get_categorical_and_numerical_features(self, X):
        # utility method to identify categorical and numerical features in the dataset
        # this can be useful for applying different preprocessing steps to different types of features (e.g., scaling numerical features, encoding categorical features)
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include=['number']).columns.tolist()
        return categorical_features, numerical_features
    

    def get_class_imbalance(self, y):
        # utility method to calculate class imbalance in the target variable
        # this can help us understand the distribution of classes and decide if we need to apply techniques like oversampling or undersampling to balance the dataset
        class_ratios = y.value_counts(normalize=True)
        return class_ratios
    

    def split_features_and_target(self, X, target_column):
        # utility method to separate features from the target variable
        # this is a common step in the machine learning pipeline before training a model
        # drop any rows from X that have missing values in the target column to ensure we have a clean dataset for training
        print(f"Original dataset shape: {X.shape}")
        X = X.dropna(subset=[target_column])
        print(f"Dataset shape after dropping rows with missing target values: {X.shape}")
        y = X[target_column]
        X = X.drop(columns=[target_column])
        return X, y
    
    def split_train_test(self, X, y, test_size=0.2, random_state=42):
        # utility method to split the dataset into training and testing sets
        # stratify on y to preserve class distribution
        X_train, X_not_train, y_train, y_not_train = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_not_train, y_not_train, test_size=0.5, random_state=random_state, stratify=y_not_train)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    