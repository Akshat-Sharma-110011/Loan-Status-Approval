import json
import mlflow
import logging
import os
import dagshub
import warnings
from src.logger import logging, section

# Suppress warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Log the start of script execution
section("MODEL REGISTRATION PROCESS", level=logging.INFO)
logging.info("Starting model registration process")

# Set up DagsHub credentials for MLflow tracking
section("ENVIRONMENT SETUP", level=logging.INFO)
logging.info("Setting up DagsHub credentials and MLflow tracking")

dagshub_token = os.getenv("MLFLOW_TRACKING_PASSWORD")
if not dagshub_token:
    logging.critical("MLFLOW_TRACKING_PASSWORD environment variable is not set")
    raise EnvironmentError("MLFLOW_TRACKING_PASSWORD environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Akshat-Sharma-110011"
repo_name = "Loan-Status-Approval"

# Set up MLflow tracking URI
tracking_uri = f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow'
logging.info(f"Setting MLflow tracking URI to: {tracking_uri}")
mlflow.set_tracking_uri(tracking_uri)

# Verify MLflow connection
try:
    client = mlflow.tracking.MlflowClient()
    logging.info("Successfully connected to MLflow tracking server")
except Exception as e:
    logging.error(f"Failed to connect to MLflow tracking server: {e}")
    raise


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    section("LOADING MODEL INFO", level=logging.INFO)
    logging.info(f"Attempting to load model info from: {file_path}")

    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)

        logging.info(f"Model info loaded successfully from {file_path}")
        logging.debug(f"Model info contents: {model_info}")

        # Log key model info details
        logging.info(f"Run ID: {model_info.get('run_id')}")
        logging.info(f"Model path: {model_info.get('model_path')}")

        return model_info

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        logging.error("Please ensure the experiment info file exists at the specified location")
        raise

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in {file_path}: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error occurred while loading the model info: {e}")
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    section("MODEL REGISTRATION", level=logging.INFO)
    logging.info(f"Starting registration process for model: {model_name}")

    try:
        # Construct model URI
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info(f"Model URI constructed: {model_uri}")

        # Register the model
        logging.info(f"Registering model {model_name} with MLflow Model Registry")
        model_version = mlflow.register_model(model_uri, model_name)
        logging.info(f"Model {model_name} registered successfully as version {model_version.version}")

        # Transition the model to "Staging" stage
        logging.info(f"Transitioning model {model_name} version {model_version.version} to 'Staging' stage")
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logging.info(f"Model {model_name} version {model_version.version} successfully transitioned to 'Staging' stage")

        return model_version.version

    except mlflow.exceptions.MlflowException as e:
        logging.error(f"MLflow error during model registration: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error during model registration: {e}")
        raise


def main():
    section("MAIN EXECUTION", level=logging.INFO)

    try:
        # Define model info path
        model_info_path = 'reports/experiment_info.json'
        logging.info(f"Using model info path: {model_info_path}")

        # Load model info
        model_info = load_model_info(model_info_path)

        # Define model name
        model_name = "my_model"
        logging.info(f"Using model name: {model_name}")

        # Register the model
        version = register_model(model_name, model_info)

        # Log successful completion
        section("REGISTRATION COMPLETED", level=logging.INFO)
        logging.info(f"Model {model_name} version {version} registration process completed successfully")

    except Exception as e:
        section("REGISTRATION FAILED", level=logging.ERROR)
        logging.error(f"Failed to complete the model registration process: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()