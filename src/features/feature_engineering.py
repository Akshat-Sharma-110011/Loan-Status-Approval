import numpy as np
import pandas as pd
import os
import joblib
import logging
from imblearn.over_sampling import SMOTE
import yaml

# Import the custom logger
from src.logger import configure_logger, SectionLogger

# Configure the logger
configure_logger()
logger = logging.getLogger(__name__)
section = SectionLogger.section


def load_processed_data(processed_train_path, processed_test_path):
    """
    Load the processed training and test data.

    Parameters:
    -----------
    processed_train_path : str
        Path to the processed training data CSV
    processed_test_path : str
        Path to the processed test data CSV

    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    section("Loading Processed Data", logger)

    try:
        # Load training data
        logger.info(f"Loading processed training data from {processed_train_path}")
        train_df = pd.read_csv(processed_train_path)
        logger.info(f"Training data loaded successfully with shape: {train_df.shape}")

        # Load test data
        logger.info(f"Loading processed test data from {processed_test_path}")
        test_df = pd.read_csv(processed_test_path)
        logger.info(f"Test data loaded successfully with shape: {test_df.shape}")

        # Extract target column
        target_column = 'loan_status'

        # Verify target column exists
        if target_column not in train_df.columns:
            logger.error(f"Target column '{target_column}' not found in training data")
            logger.info(f"Available columns: {list(train_df.columns)}")
            raise ValueError(f"Target column '{target_column}' not found in training data")

        if target_column not in test_df.columns:
            logger.error(f"Target column '{target_column}' not found in test data")
            logger.info(f"Available columns: {list(test_df.columns)}")
            raise ValueError(f"Target column '{target_column}' not found in test data")

        # Split features and target
        y_train = train_df[target_column]
        X_train = train_df.drop(target_column, axis=1)

        y_test = test_df[target_column]
        X_test = test_df.drop(target_column, axis=1)

        # Log class distribution in training data
        class_counts = y_train.value_counts()
        logger.info(f"Training data class distribution before balancing: {dict(class_counts)}")
        logger.info(f"Class imbalance ratio: 1:{class_counts[0] / class_counts[1]:.2f}")

        return X_train, y_train, X_test, y_test

    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        raise


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to handle class imbalance in the training data.

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target values
    random_state : int, default=42
        Random state for reproducibility

    Returns:
    --------
    tuple
        (X_train_balanced, y_train_balanced)
    """
    section("Applying SMOTE for Class Balancing", logger)

    try:
        # Log initial shape
        logger.info(f"Initial training data shape: {X_train.shape}")

        # Apply SMOTE
        logger.info("Initializing SMOTE")
        smote = SMOTE(random_state=random_state)

        logger.info("Applying SMOTE to balance classes")
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Log results
        logger.info(f"Balanced training data shape: {X_train_balanced.shape}")
        class_counts = pd.Series(y_train_balanced).value_counts()
        logger.info(f"Class distribution after balancing: {dict(class_counts)}")

        return X_train_balanced, y_train_balanced

    except Exception as e:
        logger.error(f"Error applying SMOTE: {str(e)}")
        raise


def save_balanced_data(X_train_balanced, y_train_balanced, X_test, y_test,
                       balanced_train_path, processed_test_path):
    """
    Save the balanced training data and the processed test data.

    Parameters:
    -----------
    X_train_balanced : pandas.DataFrame
        Balanced training features
    y_train_balanced : pandas.Series
        Balanced training target values
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target values
    balanced_train_path : str
        Path to save the balanced training data
    processed_test_path : str
        Path to save the processed test data
    """
    section("Saving Balanced Data", logger)

    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(balanced_train_path), exist_ok=True)

        # Convert balanced training data to DataFrame if necessary
        if not isinstance(X_train_balanced, pd.DataFrame):
            logger.info("Converting balanced training features to DataFrame")
            X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_test.columns)

        if not isinstance(y_train_balanced, pd.Series):
            logger.info("Converting balanced training targets to Series")
            y_train_balanced = pd.Series(y_train_balanced, name='loan_status')

        # Create full DataFrame with features and target
        train_balanced_df = X_train_balanced.copy()
        train_balanced_df['loan_status'] = y_train_balanced

        test_processed_df = X_test.copy()
        test_processed_df['loan_status'] = y_test

        # Save balanced training data
        logger.info(f"Saving balanced training data to {balanced_train_path}")
        train_balanced_df.to_csv(balanced_train_path, index=False)

        # Save processed test data (unchanged)
        logger.info(f"Saving processed test data to {processed_test_path}")
        test_processed_df.to_csv(processed_test_path, index=False)

        # Log completion
        logger.info(f"Data saved successfully")
        logger.info(f"Balanced training data shape: {train_balanced_df.shape}")
        logger.info(f"Processed test data shape: {test_processed_df.shape}")

    except Exception as e:
        logger.error(f"Error saving balanced data: {str(e)}")
        raise


def feature_engineering_pipeline(processed_train_path, processed_test_path,
                                 balanced_train_path, processed_test_output_path):
    """
    Main function that runs the entire feature engineering pipeline.

    Parameters:
    -----------
    processed_train_path : str
        Path to the processed training data
    processed_test_path : str
        Path to the processed test data
    balanced_train_path : str
        Path to save the balanced training data
    processed_test_output_path : str
        Path to save the processed test data (mainly to maintain consistency)
    """
    section("Starting Feature Engineering Pipeline", logger)

    try:
        # Load processed data
        X_train, y_train, X_test, y_test = load_processed_data(
            processed_train_path, processed_test_path
        )

        # Apply SMOTE to handle class imbalance
        X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

        # Save balanced training data and processed test data
        save_balanced_data(
            X_train_balanced, y_train_balanced, X_test, y_test,
            balanced_train_path, processed_test_output_path
        )

        section("Feature Engineering Pipeline Complete", logger)

    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Define input and output paths
    PROCESSED_TRAIN_PATH = './data/interim/train_transformed.csv'
    PROCESSED_TEST_PATH = './data/interim/test_transformed.csv'

    # Define output paths
    BALANCED_TRAIN_PATH = './data/processed/train_balanced.csv'
    PROCESSED_TEST_OUTPUT_PATH = './data/processed/test_processed.csv'

    # Run the feature engineering pipeline
    feature_engineering_pipeline(
        processed_train_path=PROCESSED_TRAIN_PATH,
        processed_test_path=PROCESSED_TEST_PATH,
        balanced_train_path=BALANCED_TRAIN_PATH,
        processed_test_output_path=PROCESSED_TEST_OUTPUT_PATH
    )