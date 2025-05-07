import numpy as np
import pandas as pd
import os
import yaml
import joblib
import json
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import the custom logger
import logging
from src.logger import configure_logger, SectionLogger

# Configure the logger
configure_logger()
logger = logging.getLogger(__name__)
section = SectionLogger.section


def load_data(train_path, test_path):
    """
    Load the balanced training data and processed test data.

    Parameters:
    -----------
    train_path : str
        Path to the balanced training data CSV
    test_path : str
        Path to the processed test data CSV

    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    section("Loading Data for Model Training", logger)

    try:
        # Load training data
        logger.info(f"Loading balanced training data from {train_path}")
        train_df = pd.read_csv(train_path)
        logger.info(f"Training data loaded successfully with shape: {train_df.shape}")

        # Load test data
        logger.info(f"Loading processed test data from {test_path}")
        test_df = pd.read_csv(test_path)
        logger.info(f"Test data loaded successfully with shape: {test_df.shape}")

        # Extract target column
        target_column = 'loan_status'

        # Verify target column exists
        if target_column not in train_df.columns:
            logger.error(f"Target column '{target_column}' not found in training data")
            logger.info(f"Available columns: {list(train_df.columns)}")
            raise ValueError(f"Target column '{target_column}' not found in training data")

        if target_column not in test_df.columns:
            logger.error(f"Target column '{target_column}' not found in test data")
            logger.info(f"Available columns: {list(test_df.columns)}")
            raise ValueError(f"Target column '{target_column}' not found in test data")

        # Split features and target
        y_train = train_df[target_column]
        X_train = train_df.drop(target_column, axis=1)

        y_test = test_df[target_column]
        X_test = test_df.drop(target_column, axis=1)

        # Log class distribution
        train_class_counts = y_train.value_counts()
        test_class_counts = y_test.value_counts()
        logger.info(f"Training data class distribution: {dict(train_class_counts)}")
        logger.info(f"Test data class distribution: {dict(test_class_counts)}")

        return X_train, y_train, X_test, y_test

    except Exception as e:
        logger.error(f"Error loading data for model training: {str(e)}")
        raise


def train_catboost_model(X_train, y_train, params=None, verbose=100):
    """
    Train a CatBoost classifier on the training data.

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target values
    params : dict, default=None
        CatBoost parameters. If None, default parameters will be used.
    verbose : int, default=100
        Verbosity of CatBoost training

    Returns:
    --------
    CatBoostClassifier
        Trained CatBoost model
    """
    section("Training CatBoost Model", logger)

    try:
        # Define default parameters if not provided
        if params is None:
            params = {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 42,
                'use_best_model': True,
                'early_stopping_rounds': 50,
            }

        # Log training parameters
        logger.info(f"CatBoost training parameters: {params}")

        # Create and train model
        logger.info("Initializing CatBoost model")
        model = CatBoostClassifier(**params)

        # Create pool for training
        train_pool = Pool(data=X_train, label=y_train)

        # Start training
        logger.info("Starting model training")
        model.fit(train_pool, verbose=verbose)

        # Get best iteration
        best_iteration = model.get_best_iteration()
        logger.info(f"Best iteration: {best_iteration}")

        # Log feature importances
        feature_importances = model.get_feature_importance()
        feature_names = X_train.columns
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Log top 10 most important features
        logger.info("Top 10 most important features:")
        for i, (feature, importance) in enumerate(zip(importance_df['Feature'][:10], importance_df['Importance'][:10])):
            logger.info(f"{i + 1}. {feature}: {importance:.4f}")

        return model

    except Exception as e:
        logger.error(f"Error training CatBoost model: {str(e)}")
        raise


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

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
        Dictionary of evaluation metrics
    """
    section("Evaluating Model Performance", logger)

    try:
        # Create test pool
        test_pool = Pool(data=X_test, label=y_test)

        # Make predictions
        logger.info("Making predictions on test data")
        y_pred_proba = model.predict_proba(test_pool)[:, 1]
        y_pred = model.predict(test_pool)

        # Calculate metrics
        metrics = {}

        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred)
        metrics['f1'] = f1_score(y_test, y_pred)
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        # Confusion matrix
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

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise


def save_model_and_metadata(model, metrics, output_dir, model_filename='model.cbm',
                            metadata_filename='model_metadata.json'):
    """
    Save the trained model and its metadata.

    Parameters:
    -----------
    model : CatBoostClassifier
        Trained CatBoost model
    metrics : dict
        Dictionary of evaluation metrics
    output_dir : str
        Directory to save the model and metadata
    model_filename : str, default='model.cbm'
        Filename for the model
    metadata_filename : str, default='model_metadata.json'
        Filename for the metadata
    """
    section("Saving Model and Metadata", logger)

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Construct full paths
        model_path = os.path.join(output_dir, model_filename)
        metadata_path = os.path.join(output_dir, metadata_filename)

        # Save model
        logger.info(f"Saving model to {model_path}")
        model.save_model(model_path)

        # Prepare metadata
        metadata = {
            'model_type': 'CatBoostClassifier',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': '1.0',
            'metrics': metrics,
            'parameters': model.get_params(),
            'features': model.feature_names_,
            'num_features': len(model.feature_names_),
            'num_trees': model.tree_count_
        }

        # Save metadata
        logger.info(f"Saving model metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info("Model and metadata saved successfully")

        return model_path, metadata_path

    except Exception as e:
        logger.error(f"Error saving model and metadata: {str(e)}")
        raise


def plot_feature_importance(model, X_train, output_dir):
    """
    Plot and save feature importance graph.

    Parameters:
    -----------
    model : CatBoostClassifier
        Trained CatBoost model
    X_train : pandas.DataFrame
        Training features to get column names
    output_dir : str
        Directory to save the plot
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get feature importances
        feature_importances = model.get_feature_importance()
        feature_names = X_train.columns

        # Create dataframe for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Take top 20 features
        top_features = importance_df.head(20)

        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(plot_path)
        logger.info(f"Feature importance plot saved to {plot_path}")

    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        logger.warning("Continuing without creating feature importance plot")


def model_building_pipeline(train_path, test_path, output_dir, model_params=None):
    """
    Main function that runs the entire model building pipeline.

    Parameters:
    -----------
    train_path : str
        Path to the balanced training data
    test_path : str
        Path to the processed test data
    output_dir : str
        Directory to save the model and metadata
    model_params : dict, default=None
        CatBoost parameters. If None, default parameters will be used.
    """
    section("Starting Model Building Pipeline", logger)

    try:
        # Load data
        X_train, y_train, X_test, y_test = load_data(train_path, test_path)

        # Train model
        model = train_catboost_model(X_train, y_train, params=model_params)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Plot feature importance
        plot_feature_importance(model, X_train, output_dir)

        # Save model and metadata
        model_path, metadata_path = save_model_and_metadata(model, metrics, output_dir)

        section("Model Building Pipeline Complete", logger)
        logger.info(f"Model saved to {model_path}")

        return model, metrics

    except Exception as e:
        logger.error(f"Error in model building pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Define input and output paths
    TRAIN_PATH = './data/processed/train_balanced.csv'
    TEST_PATH = './data/processed/test_processed.csv'
    OUTPUT_DIR = './models/model'

    # Define model parameters with optimized values from hyperparameter tuning
    model_params = {
        'iterations': 600,
        'learning_rate': 0.09069791143458081,
        'depth': 6,
        'l2_leaf_reg': 5.026670805382901,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'early_stopping_rounds': 145,
        'verbose': 100,
        'random_strength': 0.7226657998312077,
        'bagging_temperature': 4.855931236523602,
        'boosting_type': 'Plain',
        'bootstrap_type': 'MVS',
        'grow_policy': 'Depthwise',
        'leaf_estimation_method': 'Newton',
        'min_data_in_leaf': 48
    }

    # Run the model building pipeline
    model_building_pipeline(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        output_dir=OUTPUT_DIR,
        model_params=model_params
    )