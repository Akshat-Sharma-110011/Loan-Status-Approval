import numpy as np
import pandas as pd
import yaml
import cloudpickle  # Changed from joblib to cloudpickle
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

# Import the custom logger instead of using basicConfig
import logging
from src.logger import configure_logger, SectionLogger

# Configure the logger
configure_logger()
logger = logging.getLogger(__name__)
section = SectionLogger.section


# Custom transformer for IQR-based outlier removal with detailed logging
class IQROutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, factor=1.5):
        self.cols = cols
        self.factor = factor
        self.Q1 = {}
        self.Q3 = {}
        self.input_features_ = None
        logger.info(f"Initialized IQROutlierRemover with factor={factor}")

    def fit(self, X, y=None):
        logger.info(f"Fitting IQROutlierRemover on data with shape {X.shape}")
        logger.info(f"Input columns: {list(X.columns)}")

        X_ = X.copy()

        # Store input feature names for get_feature_names_out
        if isinstance(X, pd.DataFrame):
            self.input_features_ = X.columns.tolist()

        if self.cols is None:
            self.cols = X_.select_dtypes(include=['int64', 'float64']).columns.tolist()
            logger.info(f"Auto-detected numeric columns: {self.cols}")
        else:
            logger.info(f"Using specified columns: {self.cols}")

        for col in self.cols:
            self.Q1[col] = X_[col].quantile(0.25)
            self.Q3[col] = X_[col].quantile(0.75)
            logger.info(f"Column '{col}': Q1={self.Q1[col]:.4f}, Q3={self.Q3[col]:.4f}")

        return self

    def transform(self, X, y=None):
        logger.info(f"Transforming data with IQROutlierRemover, input shape: {X.shape}")
        X_ = X.copy()

        outlier_counts = {}
        for col in self.cols:
            IQR = self.Q3[col] - self.Q1[col]
            lower_bound = self.Q1[col] - (self.factor * IQR)
            upper_bound = self.Q3[col] + (self.factor * IQR)

            logger.info(f"Column '{col}': IQR={IQR:.4f}, lower_bound={lower_bound:.4f}, upper_bound={upper_bound:.4f}")

            # Count outliers before replacing
            lower_outliers = (X_[col] < lower_bound).sum()
            upper_outliers = (X_[col] > upper_bound).sum()
            outlier_counts[col] = {'lower': lower_outliers, 'upper': upper_outliers,
                                   'total': lower_outliers + upper_outliers}

            # Replace outliers with NaN
            X_[col] = np.where((X_[col] < lower_bound) | (X_[col] > upper_bound), np.nan, X_[col])

            # Log count of NaN values
            nan_count = X_[col].isna().sum()
            logger.info(
                f"Column '{col}': {lower_outliers} lower outliers, {upper_outliers} upper outliers, {nan_count} NaN values")

            # Fill NaN with median
            median_value = X_[col].median()
            X_[col] = X_[col].fillna(median_value)
            logger.info(f"Column '{col}': Filled NaN values with median {median_value:.4f}")

        logger.info(f"IQROutlierRemover transformation complete, output shape: {X_.shape}")
        logger.info(f"Outlier summary: {outlier_counts}")

        return X_

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        if input_features is not None:
            # Use provided input features
            return np.asarray(input_features, dtype=object)
        elif self.input_features_ is not None:
            # Use stored input features from fit
            return np.asarray(self.input_features_, dtype=object)
        else:
            # Fallback if no feature names available
            logger.warning("No input feature names available for IQROutlierRemover")
            return np.array([f"feature{i}" for i in range(len(self.cols))])


class LoggingTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer that logs data without modifying it"""

    def __init__(self, name):
        self.name = name
        self._feature_names = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            logger.info(f"{self.name} fit - DataFrame shape: {X.shape}, columns: {list(X.columns)}")
            self._feature_names = X.columns.tolist()
        else:
            logger.info(f"{self.name} fit - array shape: {X.shape}")
            if self._feature_names is not None:
                logger.info(f"{self.name} fit - feature names: {self._feature_names}")
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            logger.info(f"{self.name} transform - DataFrame shape: {X.shape}, columns: {list(X.columns)}")
        else:
            logger.info(f"{self.name} transform - array shape: {X.shape}")
            if self._feature_names is not None:
                logger.info(f"{self.name} transform - feature names: {self._feature_names}")
        return X

    def set_feature_names(self, names):
        self._feature_names = names
        logger.info(f"{self.name} - Set feature names: {names}")

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        if input_features is not None:
            # Just pass through the input feature names
            return np.asarray(input_features, dtype=object)
        elif self._feature_names is not None:
            # Return stored feature names if available
            return np.asarray(self._feature_names, dtype=object)
        else:
            raise ValueError(f"{self.name}: Unable to determine output feature names")


# OneHotEncoder that logs the feature names it creates
class OneHotEncoderWithLogging(OneHotEncoder):
    """OneHotEncoder that logs the feature names it creates"""

    def __init__(self, **kwargs):
        # Set drop_first=True by default, or ensure it's set if provided
        if 'drop' not in kwargs:
            kwargs['drop'] = 'first'
        super().__init__(**kwargs)
        self.feature_names_out_ = None
        self.categories_ = None

    def fit(self, X, y=None):
        result = super().fit(X, y)

        # Get and log the categories found
        if isinstance(X, pd.DataFrame):
            column_names = X.columns
        else:
            column_names = [f"col_{i}" for i in range(X.shape[1])]

        # Create and store feature names
        self.feature_names_out_ = []
        for i, (col, cats) in enumerate(zip(column_names, self.categories_)):
            # Skip the first category since drop='first'
            for cat in cats[1:]:
                self.feature_names_out_.append(f"{col}_{cat}")

        logger.info(f"OneHotEncoder created the following features:")
        for feature in self.feature_names_out_:
            logger.info(f"  - {feature}")

        return result

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_)
        return super().get_feature_names_out(input_features)


class PreprocessingPipeline:
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.feature_names_out_ = None
        self.transformers = {}

    def fit(self, X, y=None):
        # Store a copy of input data for transformations
        self.X_copy = X.copy()

        # Store original column order to ensure consistent output
        self.original_numeric_cols = [col for col in self.numeric_cols if col in X.columns]

        # Process numeric columns
        for col in self.numeric_cols:
            if col in X.columns:
                # Calculate IQR stats
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)

                # Store stats for transform
                self.transformers[col] = {
                    'Q1': Q1,
                    'Q3': Q3,
                    'median': X[col].median()
                }

                # If column is skewed, fit a power transformer
                if hasattr(self, 'skewed_cols') and col in self.skewed_cols:
                    pt = PowerTransformer(method='yeo-johnson')
                    # Reshape for sklearn compatibility
                    reshaped_data = X[[col]].copy()
                    # Handle NaN values before fitting
                    reshaped_data = reshaped_data.fillna(reshaped_data.median())
                    pt.fit(reshaped_data)
                    self.transformers[col]['power_transformer'] = pt

        # Process categorical columns with one-hot encoding
        self.one_hot_columns = []
        for col in self.categorical_cols:
            if col in X.columns:
                # Store unique values for one-hot encoding
                unique_values = sorted(X[col].unique())  # Sort for consistency
                self.transformers[col] = {
                    'categories': unique_values
                }

                # Generate expected one-hot column names - skip the first category
                for val in unique_values[1:]:  # Skip the first value (drop_first=True)
                    self.one_hot_columns.append(f"{col}_{val}")

        # Store feature names for output in the exact expected order
        self.feature_names_out_ = self.original_numeric_cols + self.one_hot_columns

        return self

    def transform(self, X):
        X_out = X.copy()

        # Process numeric columns - preserve all decimal places
        for col in self.numeric_cols:
            if col in X.columns and col in self.transformers:
                # Apply IQR outlier removal
                stats = self.transformers[col]
                Q1, Q3 = stats['Q1'], stats['Q3']
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Replace outliers with median
                outlier_mask = (X_out[col] < lower_bound) | (X_out[col] > upper_bound)
                X_out.loc[outlier_mask, col] = stats['median']

                # Apply power transformation if needed
                if 'power_transformer' in stats:
                    pt = stats['power_transformer']
                    # Handle NaN values before transforming
                    col_data = X_out[[col]].copy().fillna(stats['median'])
                    X_out[col] = pt.transform(col_data).flatten()

        # Create a new DataFrame for output to ensure proper column order
        result_df = pd.DataFrame()

        # First add all numeric columns in the original order
        for col in self.original_numeric_cols:
            if col in X_out.columns:
                result_df[col] = X_out[col]
            else:
                result_df[col] = np.nan  # Use NaN as placeholder if missing

        # Process categorical columns with one-hot encoding
        for col in self.categorical_cols:
            if col in X.columns and col in self.transformers:
                # Get the categories from the transformer
                categories = self.transformers[col]['categories']

                # Convert to categorical type with stored categories
                X_out[col] = pd.Categorical(X_out[col], categories=categories, ordered=False)

                # Create dummies with known categories and drop_first=True
                dummies = pd.get_dummies(X_out[col], prefix=col, drop_first=True, dtype=int)

                # Make sure all expected dummy columns are present (excluding the first category)
                for category in categories[1:]:  # Skip the first category (drop_first=True)
                    dummy_col = f"{col}_{category}"
                    if dummy_col in dummies.columns:
                        result_df[dummy_col] = dummies[dummy_col]
                    else:
                        result_df[dummy_col] = 0

        # Make sure all expected feature names are present
        for col in self.feature_names_out_:
            if col not in result_df.columns:
                result_df[col] = 0

        # Return only the expected columns in the right order
        return result_df[self.feature_names_out_]

    def fit_transform(self, X, y=None):
        """Combined fit and transform method"""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_)
        else:
            # Fallback
            return np.array(["feature" + str(i) for i in range(100)])


def load_feature_store(file_path='./references/feature_store.yaml'):
    """
    Load feature store YAML file and return column names for numeric and categorical features.
    """
    logger.info(f"Loading feature store from {file_path}")

    try:
        with open(file_path, 'r') as f:
            feature_store = yaml.safe_load(f)

        numeric_cols = feature_store.get('numerical_cols', [])
        categorical_cols = feature_store.get('categorical_cols', [])

        logger.info(f"Loaded numeric columns: {numeric_cols}")
        logger.info(f"Loaded categorical columns: {categorical_cols}")

        return numeric_cols, categorical_cols
    except Exception as e:
        logger.error(f"Error loading feature store: {str(e)}")
        logger.warning("Returning empty column lists")
        return [], []


def build_preprocessing_pipeline(X, target_column='loan_status'):
    """
    Build a preprocessing pipeline with better feature name handling
    """
    # Use section logger for major steps
    section("Building Preprocessing Pipeline", logger)

    # Load column types from feature_store.yaml
    numeric_cols, categorical_cols = load_feature_store('./references/feature_store.yaml')

    # Remove target column if present
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Load skewed columns
    skewed_cols = []
    try:
        with open('./references/feature_store.yaml', 'r') as f:
            config = yaml.safe_load(f)
            skewed_cols = config.get('skewed_cols', [])
    except Exception as e:
        logger.error(f"Error loading skewed columns: {str(e)}")

    # Create our custom pipeline
    preprocessor = PreprocessingPipeline(numeric_cols, categorical_cols)

    # Add skewed columns
    preprocessor.skewed_cols = skewed_cols

    return preprocessor


def process_and_save_data(preprocessor, data_path, output_path, is_train=True):
    """
    Process data using the fitted preprocessor and save it
    """
    action_type = "Training" if is_train else "Test"
    section(f"Processing {action_type} Data", logger)

    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load dataset
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")

        # Check for target column
        target_column = 'loan_status'
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in dataset")
            logger.info(f"Available columns: {list(df.columns)}")
            return

        # Extract target and features
        y = df[target_column]
        X = df.drop(target_column, axis=1)

        # Transform data (no fitting if not training data)
        if is_train:
            logger.info("Fitting and transforming training data")
            X_transformed = preprocessor.fit_transform(X)
        else:
            logger.info("Transforming test data")
            X_transformed = preprocessor.transform(X)

        # Get feature names
        feature_names = preprocessor.get_feature_names_out()

        # Convert to DataFrame
        processed_df = pd.DataFrame(X_transformed, columns=feature_names)

        # Add target column back
        processed_df[target_column] = y

        # Save processed data
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise


def create_and_save_preprocessing_pipeline(train_data_path, test_data_path, output_dir='./demo_artifacts',
                                           processed_data_dir='./demo_data/processed'):
    """
    Create and save the preprocessing pipeline, and process both train and test data
    """
    try:
        section("Starting Preprocessing Pipeline Creation", logger)

        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(processed_data_dir, exist_ok=True)

        # Load training dataset
        logger.info(f"Loading training dataset from {train_data_path}")
        train_df = pd.read_csv(train_data_path)
        logger.info(f"Training dataset loaded successfully with shape: {train_df.shape}")

        # Check for target column
        target_column = 'loan_status'
        if target_column not in train_df.columns:
            logger.error(f"Target column '{target_column}' not found in dataset")
            logger.info(f"Available columns: {list(train_df.columns)}")
            return

        # Build preprocessing pipeline
        logger.info("Building preprocessing pipeline")
        preprocessor = build_preprocessing_pipeline(train_df, target_column=target_column)

        # Prepare data for pipeline fitting (will be done in process_and_save_data for training data)
        X_train = train_df.drop(target_column, axis=1)

        # Save pipeline
        pipe_path = os.path.join(output_dir, 'preprocessing_pipeline.pkl')

        # Process and save training data - this will also fit the preprocessor
        train_processed_path = os.path.join(processed_data_dir, 'train_transformed.csv')
        process_and_save_data(preprocessor, train_data_path, train_processed_path, is_train=True)
        logger.info(f"Processed training data saved to {train_processed_path}")

        # Now save the fitted pipeline using cloudpickle instead of joblib
        with open(pipe_path, 'wb') as f:
            cloudpickle.dump(preprocessor, f)
        logger.info(f"Preprocessing pipeline saved to {pipe_path} using cloudpickle")

        # Process and save test data
        test_processed_path = os.path.join(processed_data_dir, 'test_transformed.csv')
        process_and_save_data(preprocessor, test_data_path, test_processed_path, is_train=False)
        logger.info(f"Processed test data saved to {test_processed_path}")

        section("Preprocessing Pipeline Creation and Data Processing Complete", logger)

        return preprocessor

    except Exception as e:
        logger.error(f"Error creating preprocessing pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Paths to data files - using the correct paths
    TRAIN_DATA_PATH = './data/raw/train.csv'
    TEST_DATA_PATH = './data/raw/test.csv'

    # Output directories - using the correct paths
    OUTPUT_DIR = './models/preprocessor'
    PROCESSED_DATA_DIR = './data/interim'

    # Create and save the preprocessing pipeline, and process data
    create_and_save_preprocessing_pipeline(
        train_data_path=TRAIN_DATA_PATH,
        test_data_path=TEST_DATA_PATH,
        output_dir=OUTPUT_DIR,
        processed_data_dir=PROCESSED_DATA_DIR
    )