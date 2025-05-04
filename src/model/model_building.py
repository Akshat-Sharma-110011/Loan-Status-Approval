import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import yaml
import os
from src.logger import logging, section


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    section(f"Loading Data from {file_path}", level=logging.INFO)
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s (%d rows, %d columns)', file_path, len(df), len(df.columns))
        logging.info('Memory usage: %.2f MB', df.memory_usage(deep=True).sum() / (1024 * 1024))

        # Log class distribution if last column is target
        if df.shape[1] > 0:
            target_col = df.columns[-1]
            class_dist = df[target_col].value_counts()
            logging.info('Target column "%s" distribution:\n%s', target_col, class_dist.to_dict())

        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> CatBoostClassifier:
    """Train the CatBoost Classifier."""
    section("Training CatBoost Model", level=logging.INFO)
    try:
        # Log training data information
        logging.info('Training data shape: X=%s, y=%s', X_train.shape, y_train.shape)
        logging.info('Target classes: %s', np.unique(y_train, return_counts=True))

        # Initialize model with parameters
        params = {
            'iterations': 800,
            'learning_rate': 0.16599409787390676,
            'depth': 6,
            'verbose': 0
        }
        logging.info('CatBoost parameters: %s', params)

        clf = CatBoostClassifier(**params)

        # Log start of training
        logging.info('Starting model training...')
        clf.fit(X_train, y_train)

        # Log model information
        logging.info('Model training completed successfully')
        logging.info('Feature importance calculation started')

        # Get feature importances if available
        try:
            importances = clf.get_feature_importance()
            if len(importances) > 0:
                top_n = min(10, len(importances))
                logging.info('Top %d feature importances: %s', top_n, importances[:top_n])
        except Exception as e:
            logging.warning('Could not calculate feature importances: %s', e)

        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise


def save_model(model: CatBoostClassifier, file_path: str) -> None:
    """Save the trained model to a file in CatBoost's native format."""
    section(f"Saving Model to {file_path}", level=logging.INFO)
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the model
        model.save_model(file_path)

        # Log successful save and file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logging.info('Model saved to %s (%.2f MB)', file_path, file_size_mb)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise


def main():
    section("Model Building Pipeline Started", level=logging.INFO, char='*', length=80)
    try:
        # Load training data
        train_data = load_data('./data/processed/train_balanced.csv')

        # Extract features and target
        logging.info('Extracting features and target')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        logging.info('Features extracted: X_train shape=%s, y_train shape=%s', X_train.shape, y_train.shape)

        # Train model
        model = train_model(X_train, y_train)

        # Save model in .cbm format
        save_model(model, 'models/model.cbm')

        section("Model Building Pipeline Completed Successfully", level=logging.INFO, char='*', length=80)
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        section("Model Building Pipeline Failed", level=logging.ERROR, char='!', length=80)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()