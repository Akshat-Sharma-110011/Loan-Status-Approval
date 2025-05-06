import numpy as np
import pandas as pd
import boto3

pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
from src.logger import logging, section
from src.connections import s3_connection


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    section(f"Loading Data from CSV: {data_url}", level=logging.INFO)
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s (%d rows, %d columns)', data_url, len(df), len(df.columns))
        logging.info('Memory usage: %.2f MB', df.memory_usage(deep=True).sum() / (1024 * 1024))

        # Log data info
        logging.info('Data preview: \n%s', df.head(2).to_string())
        logging.info('Data types: \n%s', df.dtypes.to_string())
        logging.info('Missing values summary: %s', df.isnull().sum().sum())

        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def fetch_data_from_s3(bucket_name: str, file_name: str) -> pd.DataFrame:
    """Fetch data from an S3 bucket using environment variables for authentication."""
    section(f"Fetching Data from S3 Bucket: {bucket_name}", level=logging.INFO)
    try:
        logging.info(f"Connecting to S3 bucket: {bucket_name}")

        # Get credentials from environment variables
        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

        if not access_key or not secret_key:
            raise ValueError("AWS credentials not found in environment variables")

        # Use boto3 to connect to S3
        s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_name)
        df = pd.read_csv(obj['Body'])  # Body is a StreamingBody

        logging.info(f"Successfully fetched data from S3 ({len(df)} rows, {len(df.columns)} columns)")
        logging.info('Memory usage: %.2f MB', df.memory_usage(deep=True).sum() / (1024 * 1024))

        return df
    except Exception as e:
        logging.error(f"Error fetching data from S3: {e}")
        raise


def try_load_from_external(file_name: str) -> pd.DataFrame:
    """Attempt to load data from ./data/external directory as fallback."""
    section(f"Attempting to Load Data from External Directory", level=logging.INFO)
    external_path = os.path.join("./data/external", file_name)

    if not os.path.exists(external_path):
        logging.error(f"Fallback file not found at {external_path}")
        raise FileNotFoundError(f"Fallback file not found at {external_path}")

    logging.info(f"Loading fallback data from {external_path}")
    return load_data(external_path)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Split the data into training and testing sets."""
    section("Splitting Data into Train and Test Sets", level=logging.INFO)
    try:
        logging.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)

        logging.info(f"Split complete - Training set: {train_data.shape}, Test set: {test_data.shape}")
        logging.info(f"Training data ratio: {len(train_data) / len(df):.2%}")
        logging.info(f"Test data ratio: {len(test_data) / len(df):.2%}")

        return train_data, test_data
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    section(f"Saving Data to {data_path}", level=logging.INFO)
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        logging.info(f"Creating directory: {raw_data_path}")
        os.makedirs(raw_data_path, exist_ok=True)

        train_file_path = os.path.join(raw_data_path, "train.csv")
        test_file_path = os.path.join(raw_data_path, "test.csv")

        logging.info(f"Saving training data ({train_data.shape}) to {train_file_path}")
        train_data.to_csv(train_file_path, index=False)

        logging.info(f"Saving test data ({test_data.shape}) to {test_file_path}")
        test_data.to_csv(test_file_path, index=False)

        # Verify files were created
        train_file_size = os.path.getsize(train_file_path) / (1024 * 1024)  # in MB
        test_file_size = os.path.getsize(test_file_path) / (1024 * 1024)  # in MB

        logging.info(f"Files saved successfully:")
        logging.info(f" - {train_file_path}: {train_file_size:.2f} MB")
        logging.info(f" - {test_file_path}: {test_file_size:.2f} MB")

    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise


def get_data(bucket_name: str, file_name: str) -> pd.DataFrame:
    """Get data with fallback mechanism: try S3 first, then local file."""
    section("Fetching Data with Fallback Support", level=logging.INFO)
    try:
        # First try S3
        logging.info("Attempting to fetch data from S3...")
        return fetch_data_from_s3(bucket_name, file_name)
    except Exception as s3_error:
        # Log S3 error
        logging.warning(f"S3 fetch failed: {s3_error}")
        logging.info("Falling back to local data source...")

        try:
            # Try loading from external directory
            return try_load_from_external(file_name)
        except Exception as local_error:
            logging.error(f"Local fallback also failed: {local_error}")
            logging.error("All data sources failed. Cannot proceed.")
            raise RuntimeError(f"Failed to fetch data from S3 and local fallback: {s3_error}; {local_error}")


def main():
    section("Data Ingestion Pipeline Started", level=logging.INFO, char='*', length=80)
    try:
        # Define parameters
        logging.info("Setting up parameters")
        test_size = 0.2
        logging.info(f"Using test_size={test_size}")

        # Parameters for data source
        bucket_name = "s3-loan-data-depository"
        file_name = "loan_data.csv"

        # Fetch data with fallback mechanism
        df = get_data(bucket_name, file_name)

        # Split data
        train_data, test_data = split_data(df, test_size=test_size, random_state=42)

        # Save data
        save_data(train_data, test_data, data_path='./data')

        section("Data Ingestion Pipeline Completed Successfully", level=logging.INFO, char='*', length=80)
    except Exception as e:
        logging.error(f"Failed to complete the data ingestion process: {e}")
        section("Data Ingestion Pipeline Failed", level=logging.ERROR, char='!', length=80)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()