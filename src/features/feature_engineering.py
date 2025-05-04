import numpy as np
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import yaml
import joblib
from src.logger import logging, section


def load_feature_store_params(params_path: str) -> dict:
    section("Loading Feature Store Configuration", level=logging.INFO)
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info('Feature config loaded from %s', params_path)
        return params
    except Exception as e:
        logging.error('Error loading feature config: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    section(f"Loading Data from {file_path}", level=logging.INFO)
    try:
        df = pd.read_csv(file_path)
        logging.info('Loaded data from %s (%d rows, %d columns)', file_path, len(df), len(df.columns))
        logging.debug('Data columns: %s', df.columns.tolist())
        return df
    except Exception as e:
        logging.error('Data loading failed: %s', e)
        raise


def apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple:
    section("Applying SMOTE Balancing", level=logging.INFO)
    try:
        # Check for NaN values before applying SMOTE
        if X.isna().any().any():
            logging.warning("Found NaN values in X. Columns with NaN: %s",
                            X.columns[X.isna().any()].tolist())
            logging.warning("NaN count per column: %s", X.isna().sum().to_dict())
            # Drop columns with NaN values
            X = X.dropna(axis=1)
            logging.info("Dropped columns with NaN. Remaining columns: %d", X.shape[1])

        logging.info("Original class distribution:\n%s", y.value_counts().to_dict())
        logging.info("Applying SMOTE (n_samples: %d, n_features: %d)", len(X), X.shape[1])
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        logging.info("SMOTE completed successfully")
        logging.info("Post-SMOTE class distribution:\n%s", pd.Series(y_res).value_counts().to_dict())
        logging.info("Post-SMOTE shape: X=%s, y=%s", X_res.shape, y_res.shape)
        return X_res, y_res
    except Exception as e:
        logging.error('SMOTE failed: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    section(f"Saving Data to {file_path}", level=logging.INFO)
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        logging.info('Created directory: %s', os.path.dirname(file_path))
        df.to_csv(file_path, index=False)
        logging.info('Saved data to %s (%d rows, %d columns)', file_path, len(df), len(df.columns))
    except Exception as e:
        logging.error('Data save failed: %s', e)
        raise


def apply_one_hot_encoding(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    section("Applying One-Hot Encoding", level=logging.INFO)
    try:
        # Log available columns for debugging
        logging.info("Available columns in dataframe: %s", df.columns.tolist())
        logging.info("Looking for categorical columns: %s", categorical_cols)

        # Check which categorical columns are actually present
        available_cat_cols = [col for col in categorical_cols if col in df.columns]
        missing_cols = set(categorical_cols) - set(available_cat_cols)

        if missing_cols:
            logging.warning("Missing categorical columns: %s", list(missing_cols))

        if not available_cat_cols:
            logging.warning("No categorical columns found in dataframe. Skipping one-hot encoding.")
            return df

        logging.info("Proceeding with encoding for available columns: %s", available_cat_cols)

        # First, standardize the categorical values (convert to lowercase)
        for col in available_cat_cols:
            if df[col].dtype == 'object':
                logging.info("Column '%s' before standardization - unique values: %s",
                             col, df[col].nunique())
                df[col] = df[col].str.strip().str.lower()
                logging.info("Column '%s' after standardization - unique values: %s",
                             col, df[col].nunique())

        # Now apply one-hot encoding
        original_shape = df.shape
        df = pd.get_dummies(df, columns=available_cat_cols, drop_first=True, dtype=int)
        logging.info("Applied one-hot encoding to: %s", available_cat_cols)
        logging.info("Shape change: %s -> %s", original_shape, df.shape)
        logging.info("New columns added: %d", df.shape[1] - original_shape[1])

        # Check for NaN values
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            logging.warning("Found NaN values in columns after encoding: %s", nan_cols)

        return df
    except Exception as e:
        logging.error("OHE failed: %s", e)
        raise


def main():
    section("Feature Engineering Pipeline Started", level=logging.INFO, char='*', length=80)
    try:
        logging.info("Loading configuration")
        params = load_feature_store_params('./references/feature_store.yaml')
        target_col = params['target_cols'][0].lower().strip()
        logging.info("Target column: %s", target_col)

        # Load data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # Print column names for both datasets for debugging
        logging.info("Train data columns: %s", train_data.columns.tolist())
        logging.info("Test data columns: %s", test_data.columns.tolist())

        # Standardize column names to lowercase
        logging.info("Standardizing column names to lowercase")
        train_data.columns = train_data.columns.str.strip().str.lower()
        test_data.columns = test_data.columns.str.strip().str.lower()
        logging.info("Column names standardized")

        # Convert categorical_cols to lowercase and validate
        categorical_cols = [col.strip().lower() for col in params['categorical_cols']]
        logging.info("Categorical columns from config: %s", categorical_cols)

        # Apply One-Hot Encoding with lowercase columns AND values
        logging.info("Applying one-hot encoding to train data")
        train_encoded = apply_one_hot_encoding(train_data.copy(), categorical_cols)

        logging.info("Applying one-hot encoding to test data")
        test_encoded = apply_one_hot_encoding(test_data.copy(), categorical_cols)

        # Extract features and target for training data
        logging.info("Separating features and target for train data")
        X_train = train_encoded.drop(columns=[target_col])
        y_train = train_encoded[target_col].astype(int)

        # Print column info before SMOTE
        logging.info("X_train shape before SMOTE: %s", X_train.shape)
        logging.info("NaN values in X_train: %s", X_train.isna().sum().sum())
        logging.info("Class distribution before SMOTE: %s", y_train.value_counts().to_dict())

        # Apply SMOTE to balance classes
        X_train_bal, y_train_bal = apply_smote(X_train, y_train)

        # Create final balanced training DataFrame
        logging.info("Creating final balanced training DataFrame")
        train_bal_df = pd.concat([
            pd.DataFrame(X_train_bal, columns=X_train.columns),
            pd.Series(y_train_bal, name=target_col)
        ], axis=1)
        logging.info("Final balanced training data shape: %s", train_bal_df.shape)

        # Save processed data
        save_data(train_bal_df, "./data/processed/train_balanced.csv")
        save_data(test_encoded, "./data/processed/test_final.csv")
        logging.info("Final test data columns: %s", test_encoded.columns.tolist())

        # Save final feature columns
        logging.info("Saving feature columns list")
        feature_cols = X_train.columns.tolist()
        joblib.dump(feature_cols, './models/feature_columns.joblib')
        logging.info("Saved %d feature columns to feature_columns.joblib", len(feature_cols))

        section("Feature Engineering Pipeline Completed Successfully", level=logging.INFO, char='*', length=80)

    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        section("Feature Engineering Pipeline Failed", level=logging.ERROR, char='!', length=80)
        raise


if __name__ == '__main__':
    main()