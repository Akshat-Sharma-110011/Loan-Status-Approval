import numpy as np
import pandas as pd
import json
import mlflow
import mlflow.sklearn
import dagshub
import os
from mlflow.exceptions import MlflowException
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.logger import logging, section

# --- Securely load credentials from environment variables ---
DAGSHUB_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
DAGSHUB_TOKEN = os.getenv("MLFLOW_TRACKING_PASSWORD")
REPO_NAME = "Loan-Status-Approval"


def initialize_mlflow():
    section("Initializing MLflow and DagsHub", level=logging.INFO)
    # Safety check
    if not DAGSHUB_USERNAME or not DAGSHUB_TOKEN:
        logging.error("DAGSHUB credentials not found in environment variables.")
        raise EnvironmentError("DAGSHUB credentials not found in environment variables.")

    try:
        # Initialize DagsHub and MLflow tracking
        logging.info(f"Initializing DagsHub with repo owner: {DAGSHUB_USERNAME}, repo name: {REPO_NAME}")
        dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)

        tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
        logging.info(f"Setting MLflow tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        EXPERIMENT_NAME = "Catboost DVC Pipeline"

        logging.info(f"Checking if experiment '{EXPERIMENT_NAME}' exists")
        if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
            logging.info(f"Creating experiment: {EXPERIMENT_NAME}")
            mlflow.create_experiment(EXPERIMENT_NAME)

        logging.info(f"Setting active experiment: {EXPERIMENT_NAME}")
        mlflow.set_experiment(EXPERIMENT_NAME)
        logging.info(f"Successfully set experiment: {EXPERIMENT_NAME}")

    except MlflowException as e:
        logging.error(f"MLflow Error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected Error during MLflow initialization: {e}")
        raise


def load_model(file_path: str):
    """Load a pre-trained model."""
    section(f"Loading Model from {file_path}", level=logging.INFO)
    try:
        model = CatBoostClassifier()
        model.load_model(file_path)  # Use CatBoost's method to load the model
        logging.info(f"Model successfully loaded from {file_path}")

        # Log model parameters
        params = model.get_params()
        logging.info(f"Model parameters: {params}")

        return model
    except FileNotFoundError:
        logging.error(f"Model file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error occurred while loading the model: {e}")
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess test data."""
    section(f"Loading Test Data from {file_path}", level=logging.INFO)
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Test data loaded successfully from {file_path}")
        logging.info(f"Test data shape: {df.shape} ({len(df)} rows, {len(df.columns)} columns)")
        logging.info(f"Test data columns: {df.columns.tolist()}")

        # Check for missing values
        missing_vals = df.isnull().sum().sum()
        if missing_vals > 0:
            logging.warning(f"Found {missing_vals} missing values in test data")

        return df
    except pd.errors.ParserError as e:
        logging.error(f"Failed to parse the CSV file: {e}")
        raise
    except FileNotFoundError:
        logging.error(f"Test data file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error occurred while loading the test data: {e}")
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model performance."""
    section("Evaluating Model Performance", level=logging.INFO)
    try:
        logging.info(f"Starting model evaluation on test data of shape {X_test.shape}")
        logging.info(f"Test target distribution: {np.unique(y_test, return_counts=True)}")

        # Model Prediction
        logging.info("Generating predictions...")
        y_pred = clf.predict(X_test)
        logging.info("Generating prediction probabilities...")
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Metrics calculation
        logging.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'auc': float(auc)
        }

        logging.info(f"Evaluation metrics:")
        for metric_name, metric_value in metrics_dict.items():
            logging.info(f"  - {metric_name}: {metric_value:.4f}")

        return metrics_dict
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save evaluation metrics to a file."""
    section(f"Saving Metrics to {file_path}", level=logging.INFO)
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        logging.info(f"Ensuring directory exists: {os.path.dirname(file_path)}")

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info(f"Metrics successfully saved to {file_path}")
        logging.info(f"Saved metrics: {metrics}")
    except Exception as e:
        logging.error(f"Error occurred while saving the metrics: {e}")
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save model info (run_id, model path) for reference."""
    section(f"Saving Model Info to {file_path}", level=logging.INFO)
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.info(f"Model info saved to {file_path}")
        logging.info(f"Run ID: {run_id}, Model path: {model_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving the model info: {e}")
        raise


def main():
    section("Model Evaluation Pipeline Started", level=logging.INFO, char='*', length=80)
    try:
        # Initialize MLflow
        initialize_mlflow()

        # Start MLflow run
        logging.info("Starting new MLflow run")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logging.info(f"MLflow run ID: {run_id}")

            # Load the model
            clf = load_model('./models/model.cbm')

            # Load test data
            test_data = load_data('./data/processed/test_final.csv')

            # Extract features and target
            section("Preparing Test Data", level=logging.INFO)
            logging.info("Extracting features and target from test data")

            if 'loan_status' in test_data.columns:
                logging.info("Using 'loan_status' as target column")
                X_test = test_data.drop('loan_status', axis=1)
                y_test = test_data['loan_status']
            else:
                logging.info("Target column 'loan_status' not found, using last column as target")
                X_test = test_data.iloc[:, :-1]
                y_test = test_data.iloc[:, -1]

            logging.info(f"Test data prepared: X_test shape={X_test.shape}, y_test shape={y_test.shape}")

            # Evaluate model
            metrics = evaluate_model(clf, X_test, y_test)
            save_metrics(metrics, 'reports/metrics.json')

            # Log results to MLflow
            section("Logging Results to MLflow", level=logging.INFO)

            # Log metrics
            logging.info("Logging metrics to MLflow")
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model parameters
            if hasattr(clf, 'get_params'):
                logging.info("Logging model parameters to MLflow")
                for param_name, param_value in clf.get_params().items():
                    mlflow.log_param(param_name, param_value)

            # Log model and artifacts
            logging.info("Logging model to MLflow")
            mlflow.sklearn.log_model(clf, "model")

            logging.info("Saving and logging model info")
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

            logging.info("Logging metrics.json as artifact")
            mlflow.log_artifact('reports/metrics.json')

            logging.info(f"MLflow run {run_id} completed successfully")

        section("Model Evaluation Pipeline Completed Successfully", level=logging.INFO, char='*', length=80)

    except Exception as e:
        logging.error(f"Failed to complete the model evaluation process: {e}")
        section("Model Evaluation Pipeline Failed", level=logging.ERROR, char='!', length=80)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()