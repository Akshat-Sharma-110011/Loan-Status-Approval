import numpy as np
import pandas as pd
import os
import yaml
from sklearn.preprocessing import PowerTransformer
from src.logger import logging


# Load feature groups from YAML
def load_feature_config(yaml_path='./references/feature_store.yaml'):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info("Feature configuration loaded from YAML")
    return config


def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = df.shape[0]
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        after = df.shape[0]
        logging.info(f'Removed outliers from "{col}": {before - after} rows dropped')
    return df


def preprocess_dataframe(df, config, dataset_name='dataset'):
    logging.info(f"Starting preprocessing for {dataset_name}...")

    # Get columns from config
    categorical_cols = config['categorical_cols']
    skewed_cols = config['skewed_cols']

    logging.info(f"Initial shape: {df.shape}")

    # Drop missing values
    df = df.dropna()
    logging.info(f"After dropping missing values: {df.shape}")

    # Drop duplicates
    df = df.drop_duplicates()
    logging.info(f"After dropping duplicates: {df.shape}")

    # Remove outliers from skewed columns
    df = remove_outliers_iqr(df, skewed_cols)
    logging.info(f"After outlier removal: {df.shape}")

    # Apply Yeo-Johnson Transformation
    pt = PowerTransformer(method='yeo-johnson')
    df[skewed_cols] = pt.fit_transform(df[skewed_cols])
    logging.info(f"Applied Yeo-Johnson to: {skewed_cols}")

    # One-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    logging.info(f"One-hot encoded categorical columns: {categorical_cols}")
    logging.info(f"Final shape of {dataset_name}: {df.shape}")

    return df


def main():
    try:
        config = load_feature_config()

        # Load raw data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info("Loaded train and test datasets")

        # Preprocess data
        train_processed = preprocess_dataframe(train_data, config, 'train')
        test_processed = preprocess_dataframe(test_data, config, 'test')

        # Save to interim
        output_path = './data/interim'
        os.makedirs(output_path, exist_ok=True)
        train_processed.to_csv(os.path.join(output_path, 'train_processed.csv'), index=False)
        test_processed.to_csv(os.path.join(output_path, 'test_processed.csv'), index=False)
        logging.info(f"Processed files saved to {output_path}")

    except Exception as e:
        logging.error("Preprocessing failed: %s", e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()