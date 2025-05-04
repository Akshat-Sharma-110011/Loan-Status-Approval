import numpy as np
import pandas as pd
import os
import yaml
import joblib
import time
from datetime import datetime
from sklearn.preprocessing import PowerTransformer
from src.logger import logging, section


def load_feature_config(yaml_path='./references/feature_store.yaml'):
    """Load feature configuration from YAML file"""
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Feature configuration loaded successfully from {yaml_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load feature configuration: {str(e)}")
        raise


def remove_outliers_iqr(df, columns, iqr_thresholds=None):
    """Remove outliers using IQR method"""
    section("OUTLIER REMOVAL")

    thresholds = {}
    total_removed = 0
    initial_size = df.shape[0]

    for col in columns:
        logging.info(f"Processing outliers for column: {col}")

        if iqr_thresholds and col in iqr_thresholds:
            lower, upper = iqr_thresholds[col]
            logging.info(f"Using predefined thresholds for {col}: [{lower:.3f}, {upper:.3f}]")
        else:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            thresholds[col] = (lower, upper)
            logging.info(f"Calculated thresholds for {col}: [{lower:.3f}, {upper:.3f}]")

        before = df.shape[0]
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        removed = before - df.shape[0]
        total_removed += removed

        if removed > 0:
            logging.info(f"Removed {removed} outliers from '{col}' ({removed / before * 100:.2f}% of data)")
        else:
            logging.info(f"No outliers found in '{col}'")

    if total_removed > 0:
        removal_percentage = (total_removed / initial_size) * 100
        logging.info(f"Total outliers removed: {total_removed} ({removal_percentage:.2f}% of original data)")
    else:
        logging.info("No outliers were removed from any columns")

    return df, thresholds


def preprocess_dataframe(df, config, dataset_name='dataset', mode='train', transformers=None):
    """Preprocess the dataframe with cleaning, outlier removal and transformations"""
    section(f"PREPROCESSING {dataset_name.upper()}")
    start_time = time.time()

    skewed_cols = config['skewed_cols']
    logging.info(f"Initial shape: {df.shape} ({df.memory_usage().sum() / (1024 ** 2):.2f} MB)")
    logging.info(f"Processing {len(skewed_cols)} skewed columns: {', '.join(skewed_cols)}")

    # Step 1: Remove duplicates and NaN values
    initial_count = df.shape[0]
    df_no_na = df.dropna()
    na_removed = initial_count - df_no_na.shape[0]

    df_clean = df_no_na.drop_duplicates()
    dupes_removed = df_no_na.shape[0] - df_clean.shape[0]

    df = df_clean

    if na_removed > 0:
        logging.info(f"Removed {na_removed} rows with NaN values ({na_removed / initial_count * 100:.2f}%)")
    else:
        logging.info("No NaN values found in the dataset")

    if dupes_removed > 0:
        logging.info(f"Removed {dupes_removed} duplicate rows ({dupes_removed / initial_count * 100:.2f}%)")
    else:
        logging.info("No duplicate rows found in the dataset")

    logging.info(f"After cleaning: {df.shape}")

    # Step 2: Remove outliers
    df, outlier_thresholds = remove_outliers_iqr(df, skewed_cols, config.get('iqr_thresholds'))

    # Step 3: Apply power transformation to skewed columns
    section("FEATURE TRANSFORMATION")

    if mode == 'train':
        logging.info("Training mode: Fitting new power transformers")
        transformers = {}
        for col in skewed_cols:
            skewness_before = df[col].skew()
            logging.info(f"Transforming '{col}' (skewness before: {skewness_before:.4f})")

            pt = PowerTransformer(method='yeo-johnson')
            transformed_values = pt.fit_transform(df[[col]]).flatten().astype(np.float32)
            df[col] = transformed_values

            skewness_after = df[col].skew()
            transformers[col] = pt

            logging.info(f"✓ '{col}' transformation complete (skewness after: {skewness_after:.4f})")

        # Save transformers
        transformer_path = './models/power_transformers.joblib'
        joblib.dump(transformers, transformer_path)
        logging.info(f"Saved transformers to {transformer_path}")
    else:
        logging.info("Test mode: Using pre-trained transformers")
        for col in skewed_cols:
            if col in transformers:
                skewness_before = df[col].skew()
                logging.info(f"Applying transformer to '{col}' (skewness before: {skewness_before:.4f})")

                pt = transformers[col]
                df[col] = pt.transform(df[[col]]).flatten().astype(np.float32)

                skewness_after = df[col].skew()
                logging.info(f"✓ '{col}' transformation applied (skewness after: {skewness_after:.4f})")
            else:
                logging.warning(f"No transformer found for column '{col}', skipping")

    elapsed_time = time.time() - start_time
    logging.info(f"Final shape of {dataset_name}: {df.shape} ({elapsed_time:.2f} seconds)")
    return df, transformers


def main():
    """Main function to execute the preprocessing pipeline"""
    start_time = time.time()
    section("DATA PREPROCESSING PIPELINE", level=logging.INFO, char='*', length=80)

    try:
        # Step 1: Load configuration
        logging.info("Starting preprocessing pipeline")
        config = load_feature_config()
        os.makedirs('./models', exist_ok=True)
        logging.info("Models directory created/confirmed")

        # Step 2: Load raw data
        section("DATA LOADING")
        start_load = time.time()

        train_path = './data/raw/train.csv'
        test_path = './data/raw/test.csv'

        logging.info(f"Loading training data from {train_path}")
        train_data = pd.read_csv(train_path)
        logging.info(f"Training data loaded: {train_data.shape[0]:,} rows, {train_data.shape[1]:,} columns")

        logging.info(f"Loading test data from {test_path}")
        test_data = pd.read_csv(test_path)
        logging.info(f"Test data loaded: {test_data.shape[0]:,} rows, {test_data.shape[1]:,} columns")

        load_time = time.time() - start_load
        logging.info(f"Data loading completed in {load_time:.2f} seconds")

        # Log column information
        train_cols = train_data.columns.tolist()
        test_cols = test_data.columns.tolist()

        # Find differences in columns
        train_only = set(train_cols) - set(test_cols)
        test_only = set(test_cols) - set(train_cols)

        logging.info(f"Train data columns: {', '.join(train_cols)}")
        logging.info(f"Test data columns: {', '.join(test_cols)}")

        if train_only:
            logging.warning(f"Columns only in training data: {', '.join(train_only)}")
        if test_only:
            logging.warning(f"Columns only in test data: {', '.join(test_only)}")

        # Step 3: Preprocess train data
        section("TRAIN DATA PREPROCESSING")
        train_processed, transformers = preprocess_dataframe(
            train_data, config, 'train', mode='train'
        )

        # Step 4: Save feature column names from train set
        feature_cols = train_processed.columns.tolist()
        feature_cols_path = './models/feature_columns.joblib'
        joblib.dump(feature_cols, feature_cols_path)
        logging.info(f"Saved {len(feature_cols)} feature columns to {feature_cols_path}")

        # Step 5: Preprocess test data
        section("TEST DATA PREPROCESSING")
        transformers = joblib.load('./models/power_transformers.joblib')
        test_processed, _ = preprocess_dataframe(
            test_data, config, 'test', mode='test',
            transformers=transformers
        )

        # Step 6: Align test columns with train
        section("COLUMN ALIGNMENT")
        feature_cols = joblib.load('./models/feature_columns.joblib')
        missing_cols = set(feature_cols) - set(test_processed.columns)

        if missing_cols:
            logging.warning(f"Adding {len(missing_cols)} missing columns to test data: {', '.join(missing_cols)}")
            for col in missing_cols:
                test_processed[col] = 0

        # Ensure column order matches training data
        test_processed = test_processed[feature_cols]
        logging.info(f"Test data columns aligned with training data: {test_processed.shape[1]} columns")

        # Step 7: Save processed data
        section("SAVING PROCESSED DATA")
        output_path = './data/interim'
        os.makedirs(output_path, exist_ok=True)

        train_output = os.path.join(output_path, 'train_processed.csv')
        test_output = os.path.join(output_path, 'test_processed.csv')

        train_processed.to_csv(train_output, index=False)
        logging.info(f"Saved processed training data to {train_output}")

        test_processed.to_csv(test_output, index=False)
        logging.info(f"Saved processed test data to {test_output}")

        # Step 8: Summarize
        total_time = time.time() - start_time
        section("PREPROCESSING COMPLETE", char='*', length=80)
        logging.info(f"Total preprocessing time: {total_time:.2f} seconds")
        logging.info(f"Processed files saved to {output_path}")

    except Exception as e:
        section("ERROR ENCOUNTERED", level=logging.ERROR, char='!', length=80)
        logging.error(f"Preprocessing failed: {str(e)}")
        import traceback
        logging.error(f"Error details:\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()