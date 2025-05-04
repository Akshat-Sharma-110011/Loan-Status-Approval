import numpy as np
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import yaml
from src.logger import logging


def load_feature_store_params(params_path: str) -> dict:
    """Load feature store parameters (such as target column) from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Feature store parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Apply SMOTE to balance the classes in the training set."""
    try:
        logging.info("Applying SMOTE to balance the classes...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logging.info("SMOTE applied successfully.")
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)
    except Exception as e:
        logging.error('Error during SMOTE resampling: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        # Load parameters from the feature_store.yaml file
        feature_store_params = load_feature_store_params('./references/feature_store.yaml')

        # Extract the target column and other feature categories
        target_cols = feature_store_params['target_cols']
        target_col = target_cols[0] if target_cols else None

        if target_col is None:
            logging.error('No target column defined in the feature_store.yaml')
            raise ValueError("Target column not found in the feature_store.yaml.")

        categorical_cols = feature_store_params.get('categorical_cols', [])
        numerical_cols = feature_store_params.get('numerical_cols', [])
        skewed_cols = feature_store_params.get('skewed_cols', [])
        norm_cols = feature_store_params.get('norm_cols', [])

        # Log the columns being used
        logging.info(f"Using target column: {target_col}")
        logging.info(f"Categorical columns: {categorical_cols}")
        logging.info(f"Numerical columns: {numerical_cols}")
        logging.info(f"Skewed columns: {skewed_cols}")
        logging.info(f"Normalized columns: {norm_cols}")

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # Standardize column names (if necessary)
        train_data.columns = train_data.columns.str.strip().str.lower()

        # Check if the target column exists
        if target_col.lower() in train_data.columns:
            X_train = train_data.drop(columns=[target_col.lower()])
            y_train = train_data[target_col.lower()]
        else:
            logging.error(f"Target column '{target_col}' not found in the data.")
            raise ValueError(f"Target column '{target_col}' not found.")

        # You can use categorical_cols, numerical_cols, etc., for further processing here

        X_train_bal, y_train_bal = apply_smote(X_train, y_train)

        train_bal_df = pd.concat([X_train_bal, y_train_bal], axis=1)

        save_data(train_bal_df, os.path.join("./data", "processed", "train_balanced.csv"))
        save_data(test_data, os.path.join("./data", "processed", "test.csv"))
    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()