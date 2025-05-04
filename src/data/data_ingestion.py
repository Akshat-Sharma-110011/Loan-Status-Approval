import numpy as np
import pandas as pd
import boto3
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
from src.logger import logging, section
from src.connections import s3_connection


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    section("Loading Parameters", level=logging.INFO)
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info('Parameters retrieved from %s', params_path)
        logging.debug('Parameter contents: %s', params)
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


def fetch_data_from_s3(bucket_name: str, access_key: str, secret_key: str, file_name: str) -> pd.DataFrame:
    """Fetch data from an S3 bucket."""
    section(f"Fetching Data from S3 Bucket: {bucket_name}", level=logging.INFO)
    try:
        logging.info(f"Connecting to S3 bucket: {bucket_name}")

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
        test_file_path = os.path.join(raw_data_path, "test_final.csv")

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


def main():
    section("Data Ingestion Pipeline Started", level=logging.INFO, char='*', length=80)
    try:
        # Define parameters
        logging.info("Setting up parameters")
        # params = load_params(params_path='params.yaml')
        # test_size = params['data_ingestion']['test_size']
        test_size = 0.2
        logging.info(f"Using test_size={test_size}")

        # Fetch data from S3
        logging.info("Fetching data from S3")
        bucket_name = "s3-loan-data-depository"
        access_key = "AWS_ACCESS_KEY_ID"
        secret_key = "AWS_SECRET_ACCESS_KEY"
        file_name = "loan_data.csv"

        # Note: In production code, these credentials should be stored securely
        # and not hardcoded in the script
        df = fetch_data_from_s3(bucket_name, access_key, secret_key, file_name)

        # Alternative local data loading
        # df = load_data('notebooks/loan_data.csv')

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