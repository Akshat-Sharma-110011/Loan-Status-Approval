import numpy as np
import pandas as pd
import json
import mlflow
import mlflow.catboost
import dagshub
import os
from mlflow.exceptions import MlflowException
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from datetime import datetime

# Import the custom logger
import logging
from src.logger import configure_logger, SectionLogger

# Configure the logger
configure_logger()
logger = logging.getLogger(__name__)
section = SectionLogger.section

# --- Securely load credentials from environment variables ---
DAGSHUB_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
DAGSHUB_TOKEN = os.getenv("MLFLOW_TRACKING_PASSWORD")
REPO_NAME = "Loan-Status-Approval"


def initialize_mlflow():
    """
    Initialize MLflow tracking with DagsHub integration.
    """
    section("Initializing MLflow and DagsHub", logger)

    # Safety check for credentials
    if not DAGSHUB_USERNAME or not DAGSHUB_TOKEN:
        logger.error("DAGSHUB credentials not found in environment variables.")
        raise EnvironmentError("DAGSHUB credentials not found in environment variables.")

    try:
        # Initialize DagsHub and MLflow tracking
        logger.info(f"Initializing DagsHub with repo owner: {DAGSHUB_USERNAME}, repo name: {REPO_NAME}")
        dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)

        tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
        logger.info(f"Setting MLflow tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        EXPERIMENT_NAME = "Loan Approval Prediction"

        logger.info(f"Checking if experiment '{EXPERIMENT_NAME}' exists")
        if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
            logger.info(f"Creating experiment: {EXPERIMENT_NAME}")
            mlflow.create_experiment(EXPERIMENT_NAME)

        logger.info(f"Setting active experiment: {EXPERIMENT_NAME}")
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"Successfully set experiment: {EXPERIMENT_NAME}")

    except MlflowException as e:
        logger.error(f"MLflow Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected Error during MLflow initialization: {e}")
        raise


def load_model(model_path):
    """
    Load a pre-trained CatBoost model.

    Parameters:
    -----------
    model_path : str
        Path to the saved model file

    Returns:
    --------
    CatBoostClassifier
        Loaded CatBoost model
    """
    section(f"Loading Model from {model_path}", logger)

    try:
        # Ensure the path ends with the model file if it doesn't already
        full_model_path = model_path
        if not model_path.endswith('.cbm'):
            full_model_path = os.path.join(model_path, 'model.cbm')

        # Load the CatBoost model
        model = CatBoostClassifier()
        model.load_model(full_model_path)
        logger.info(f"Model successfully loaded from {full_model_path}")

        # Log model parameters
        params = model.get_params()
        logger.info(f"Model parameters: {params}")

        # Log model metadata if available
        metadata_path = os.path.join(os.path.dirname(full_model_path), 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Model metadata loaded from {metadata_path}")
            logger.info(f"Model training metrics: {metadata.get('metrics', 'N/A')}")
        else:
            logger.warning(f"Model metadata file not found at {metadata_path}")

        return model

    except FileNotFoundError:
        logger.error(f"Model file not found: {full_model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def load_test_data(test_path):
    """
    Load the processed test data.

    Parameters:
    -----------
    test_path : str
        Path to the processed test data CSV

    Returns:
    --------
    tuple
        (X_test, y_test) - Features and target variables
    """
    section(f"Loading Test Data from {test_path}", logger)

    try:
        # Load test data
        logger.info(f"Loading processed test data from {test_path}")
        test_df = pd.read_csv(test_path)
        logger.info(f"Test data loaded successfully with shape: {test_df.shape}")

        # Check for missing values
        missing_vals = test_df.isnull().sum().sum()
        if missing_vals > 0:
            logger.warning(f"Found {missing_vals} missing values in test data")

        # Extract target column
        target_column = 'loan_status'

        # Verify target column exists
        if target_column not in test_df.columns:
            logger.error(f"Target column '{target_column}' not found in test data")
            logger.info(f"Available columns: {list(test_df.columns)}")
            raise ValueError(f"Target column '{target_column}' not found in test data")

        # Split features and target
        y_test = test_df[target_column]
        X_test = test_df.drop(target_column, axis=1)

        # Log class distribution
        test_class_counts = y_test.value_counts()
        logger.info(f"Test data class distribution: {dict(test_class_counts)}")

        return X_test, y_test

    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise


def validate_features(model, X_test):
    """
    Validate that test data features match model features.

    Parameters:
    -----------
    model : CatBoostClassifier
        Trained CatBoost model
    X_test : pandas.DataFrame
        Test features

    Returns:
    --------
    pandas.DataFrame
        Validated and adjusted test features
    """
    section("Validating Feature Compatibility", logger)

    try:
        # Get model features
        model_features = model.feature_names_
        test_columns = X_test.columns.tolist()

        logger.info(f"Model expects {len(model_features)} features")
        logger.info(f"Test data has {len(test_columns)} features")

        # Check for missing features
        missing_features = set(model_features) - set(test_columns)
        if missing_features:
            logger.warning(f"Missing features in test data: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                X_test[feature] = 0
            logger.info(f"Added {len(missing_features)} missing features to test data with default value 0")

        # Check for extra features
        extra_features = set(test_columns) - set(model_features)
        if extra_features:
            logger.warning(f"Extra features in test data: {extra_features}")
            # Remove extra features
            X_test = X_test.drop(columns=list(extra_features))
            logger.info(f"Removed {len(extra_features)} extra features from test data")

        # Ensure column order matches model's expected order
        X_test = X_test[model_features]
        logger.info("Test data columns reordered to match model's expected feature order")

        return X_test

    except Exception as e:
        logger.error(f"Error validating features: {str(e)}")
        raise


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and compute performance metrics.

    Parameters:
    -----------
    model : CatBoostClassifier
        Trained CatBoost model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target values

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    section("Evaluating Model Performance", logger)

    try:
        # Create test pool
        test_pool = Pool(data=X_test, label=y_test)

        # Generate predictions
        logger.info("Generating predictions on test data")
        y_pred_proba = model.predict_proba(test_pool)[:, 1]
        y_pred = model.predict(test_pool)

        # Calculate evaluation metrics
        metrics = {}

        # Classification metrics
        metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
        metrics['precision'] = float(precision_score(y_test, y_pred))
        metrics['recall'] = float(recall_score(y_test, y_pred))
        metrics['f1'] = float(f1_score(y_test, y_pred))
        metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))

        # Log confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Log evaluation metrics
        logger.info("Model performance metrics:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")

        # Log confusion matrix
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        # Log classification report
        class_report = classification_report(y_test, y_pred)
        logger.info(f"Classification Report:\n{class_report}")

        return metrics, y_pred, y_pred_proba

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise


def generate_evaluation_plots(y_test, y_pred, y_pred_proba, output_dir):
    """
    Generate and save evaluation plots.

    Parameters:
    -----------
    y_test : pandas.Series
        Test target values
    y_pred : numpy.ndarray
        Predicted class labels
    y_pred_proba : numpy.ndarray
        Predicted class probabilities
    output_dir : str
        Directory to save plots
    """
    section("Generating Evaluation Plots", logger)

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'figures/confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion matrix plot saved to {cm_path}")

        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.tight_layout()
        roc_path = os.path.join(output_dir, 'figures/roc_curve.png')
        plt.savefig(roc_path)
        plt.close()
        logger.info(f"ROC curve plot saved to {roc_path}")

        # 3. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.tight_layout()
        pr_path = os.path.join(output_dir, 'figures/precision_recall_curve.png')
        plt.savefig(pr_path)
        plt.close()
        logger.info(f"Precision-Recall curve plot saved to {pr_path}")

        # 4. Prediction Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(y_pred_proba, bins=50, kde=True)
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Probabilities')
        plt.tight_layout()
        dist_path = os.path.join(output_dir, 'figures/prediction_distribution.png')
        plt.savefig(dist_path)
        plt.close()
        logger.info(f"Prediction distribution plot saved to {dist_path}")

        # Return paths to all generated plots
        return {
            'confusion_matrix': cm_path,
            'roc_curve': roc_path,
            'precision_recall_curve': pr_path,
            'prediction_distribution': dist_path
        }

    except Exception as e:
        logger.error(f"Error generating evaluation plots: {str(e)}")
        logger.warning("Continuing without creating some plots")
        return {}


def save_results(metrics, output_dir, model_info=None):
    """
    Save evaluation metrics and model information to files.

    Parameters:
    -----------
    metrics : dict
        Dictionary of evaluation metrics
    output_dir : str
        Directory to save results
    model_info : dict, optional
        Model information to save
    """
    section("Saving Evaluation Results", logger)

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics to JSON file
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved to {metrics_path}")

        # Save model info if provided
        if model_info:
            info_path = os.path.join(output_dir, 'model_evaluation_info.json')
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            logger.info(f"Model evaluation info saved to {info_path}")

        return metrics_path

    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def model_evaluation_pipeline(model_path, test_path, output_dir):
    """
    Main function that runs the entire model evaluation pipeline.

    Parameters:
    -----------
    model_path : str
        Path to the trained model file
    test_path : str
        Path to the processed test data
    output_dir : str
        Directory to save evaluation results
    """
    section("Starting Model Evaluation Pipeline", logger, char='*', length=80)

    try:
        # Initialize MLflow if credentials are available
        if DAGSHUB_USERNAME and DAGSHUB_TOKEN:
            initialize_mlflow()
            mlflow_tracking = True
        else:
            logger.warning("MLflow tracking credentials not available. Results will be saved locally only.")
            mlflow_tracking = False

        # Start MLflow run if tracking is enabled
        run_id = None
        if mlflow_tracking:
            mlflow_run = mlflow.start_run()
            run_id = mlflow_run.info.run_id
            logger.info(f"MLflow run started with ID: {run_id}")

        # Load the model
        # Store the original model_path without .cbm suffix for the info file
        original_model_path = model_path
        if model_path.endswith('.cbm'):
            original_model_path = os.path.dirname(model_path)

        model = load_model(model_path)

        # Load and prepare test data
        X_test, y_test = load_test_data(test_path)

        # Validate features
        X_test = validate_features(model, X_test)

        # Evaluate model performance
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

        # Generate evaluation plots
        plot_paths = generate_evaluation_plots(y_test, y_pred, y_pred_proba, output_dir)

        # Save evaluation results with the proper model_path format
        model_info = {
            'model_path': original_model_path,
            'test_path': test_path,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mlflow_run_id': run_id
        }
        metrics_path = save_results(metrics, output_dir, model_info)

        # Log to MLflow if tracking is enabled
        if mlflow_tracking:
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log parameters
            model_params = model.get_params()
            for key, value in model_params.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)

            # Log model
            mlflow.catboost.log_model(model, "model")

            # Log artifacts
            mlflow.log_artifact(metrics_path)
            for plot_type, plot_path in plot_paths.items():
                mlflow.log_artifact(plot_path)

            # End the MLflow run
            mlflow.end_run()
            logger.info(f"MLflow run {run_id} completed successfully")

        section("Model Evaluation Pipeline Complete", logger, char='*', length=80)
        return metrics

    except Exception as e:
        logger.error(f"Error in model evaluation pipeline: {str(e)}")
        if mlflow_tracking and mlflow.active_run():
            mlflow.end_run()
        raise


if __name__ == "__main__":
    # Define paths
    MODEL_PATH = './models/model'  # Removed the '/model.cbm' suffix
    TEST_PATH = './data/processed/test_processed.csv'  # Updated path to match example
    OUTPUT_DIR = './reports'

    # Run the model evaluation pipeline
    model_evaluation_pipeline(
        model_path=MODEL_PATH,
        test_path=TEST_PATH,
        output_dir=OUTPUT_DIR
    )