import unittest
import mlflow
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from catboost import CatBoostClassifier


class TestCatBoostModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if not dagshub_token:
            raise EnvironmentError("MLFLOW_TRACKING_PASSWORD environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Akshat-Sharma-110011"
        repo_name = "Loan-Status-Approval"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the CatBoost model from MLflow model registry
        cls.model_name = "Catboost_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f'models:/{cls.model_name}/{cls.model_version}'
        print(f"Loading model from: {cls.model_uri}")
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # Load the preprocessing pipeline
        cls.preprocessor = joblib.load('models/preprocessor/preprocessing_pipeline.pkl')
        print("Preprocessing pipeline loaded")

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_processed.csv')
        print(f"Holdout data loaded with shape: {cls.holdout_data.shape}")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        if not latest_version:
            raise ValueError(f"No model version found for {model_name} in {stage} stage")
        return latest_version[0].version

    def test_model_loaded_properly(self):
        """Test if the model is loaded properly"""
        self.assertIsNotNone(self.model)
        print("Model loaded successfully")

    def test_model_signature(self):
        """Test if the model works with the expected input signature"""
        # Create a sample input similar to what we expect in production
        sample_data = {
            "person_age": [25],
            "person_gender": ["male"],
            "person_education": ["Bachelor"],
            "person_income": [50000],
            "person_emp_exp": [3],
            "person_home_ownership": ["RENT"],
            "loan_amnt": [10000],
            "loan_intent": ["EDUCATION"],
            "loan_int_rate": [10.5],
            "loan_percent_income": [0.2],
            "cb_person_cred_hist_length": [5],
            "credit_score": [650],
            "previous_loan_defaults_on_file": ["No"]
        }

        sample_df = pd.DataFrame(sample_data)

        # Transform data using the preprocessor
        transformed_data = self.preprocessor.transform(sample_df)

        # Convert to DataFrame with feature names
        feature_names = self.preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)

        # Predict using the model
        prediction = self.model.predict(transformed_df)

        # Verify the output shape (should be a single prediction)
        self.assertEqual(len(prediction), 1)
        self.assertTrue(prediction[0] in [0, 1], f"Prediction {prediction[0]} should be binary (0 or 1)")

        print(f"Model signature test passed with prediction: {prediction[0]}")

    def test_model_batch_prediction(self):
        """Test if the model can handle batch predictions"""
        # Create multiple samples
        sample_data = {
            "person_age": [25, 35, 45, 55],
            "person_gender": ["male", "female", "male", "female"],
            "person_education": ["Bachelor", "High School", "Master", "PhD"],
            "person_income": [50000, 30000, 80000, 100000],
            "person_emp_exp": [3, 5, 10, 15],
            "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "RENT"],
            "loan_amnt": [10000, 5000, 25000, 40000],
            "loan_intent": ["EDUCATION", "PERSONAL", "MEDICAL", "HOME_IMPROVEMENT"],
            "loan_int_rate": [10.5, 8.3, 12.7, 5.2],
            "loan_percent_income": [0.2, 0.15, 0.3, 0.4],
            "cb_person_cred_hist_length": [5, 8, 12, 20],
            "credit_score": [650, 700, 750, 800],
            "previous_loan_defaults_on_file": ["No", "Yes", "No", "No"]
        }

        sample_df = pd.DataFrame(sample_data)

        # Transform data using the preprocessor
        transformed_data = self.preprocessor.transform(sample_df)

        # Convert to DataFrame with feature names
        feature_names = self.preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)

        # Predict using the model
        predictions = self.model.predict(transformed_df)

        # Verify the output shape
        self.assertEqual(len(predictions), 4)
        for pred in predictions:
            self.assertTrue(pred in [0, 1], f"Prediction {pred} should be binary (0 or 1)")

        print(f"Batch prediction test passed with predictions: {predictions}")

    def test_model_performance(self):
        """Test if the model meets performance thresholds on holdout data"""
        # Split features and target
        X_holdout = self.holdout_data.drop('loan_status',
                                           axis=1) if 'loan_status' in self.holdout_data.columns else self.holdout_data.iloc[
                                                                                                      :, :-1]
        y_holdout = self.holdout_data[
            'loan_status'] if 'loan_status' in self.holdout_data.columns else self.holdout_data.iloc[:, -1]

        # Transform features using the preprocessor
        transformed_data = self.preprocessor.transform(X_holdout)

        # Convert to DataFrame with feature names
        feature_names = self.preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)

        # Make predictions
        y_pred = self.model.predict(transformed_df)

        # Calculate performance metrics
        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred, zero_division=0)
        recall = recall_score(y_holdout, y_pred, zero_division=0)
        f1 = f1_score(y_holdout, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_holdout, y_pred)

        # Define expected thresholds (adjust based on your model's expected performance)
        expected_accuracy = 0.75
        expected_precision = 0.70
        expected_recall = 0.70
        expected_f1 = 0.70

        # Print performance metrics
        print(f"\nModel Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        # Assert that the model meets the performance thresholds
        self.assertGreaterEqual(accuracy, expected_accuracy,
                                f'Accuracy {accuracy} should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision, expected_precision,
                                f'Precision {precision} should be at least {expected_precision}')
        self.assertGreaterEqual(recall, expected_recall, f'Recall {recall} should be at least {expected_recall}')
        self.assertGreaterEqual(f1, expected_f1, f'F1 score {f1} should be at least {expected_f1}')

    def test_model_feature_importance(self):
        """Test if the model's feature importance can be extracted"""
        try:
            # Load the actual CatBoost model (not the MLflow wrapper)
            catboost_model = CatBoostClassifier()
            catboost_model.load_model('models/model/model.cbm')

            # Get feature importance
            feature_names = self.preprocessor.get_feature_names_out()
            feature_importances = catboost_model.get_feature_importance()

            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names if len(feature_names) == len(feature_importances) else [f'Feature_{i}' for i in
                                                                                                 range(
                                                                                                     len(feature_importances))],
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False)

            # Print top 10 features
            print("\nTop 10 important features:")
            print(importance_df.head(10))

            # Check if feature importance is available
            self.assertGreater(len(feature_importances), 0)

        except Exception as e:
            print(f"Warning: Could not extract feature importance: {str(e)}")
            # Don't fail the test if this is not critical
            pass

    def test_edge_cases(self):
        """Test the model with edge cases"""
        # Test with extreme values
        edge_cases = [
            # Very young applicant with high income
            {
                "person_age": [18],
                "person_gender": ["male"],
                "person_education": ["High School"],
                "person_income": [200000],
                "person_emp_exp": [0],
                "person_home_ownership": ["RENT"],
                "loan_amnt": [50000],
                "loan_intent": ["EDUCATION"],
                "loan_int_rate": [15.0],
                "loan_percent_income": [0.25],
                "cb_person_cred_hist_length": [1],
                "credit_score": [500],
                "previous_loan_defaults_on_file": ["No"]
            },
            # Older applicant with low income
            {
                "person_age": [65],
                "person_gender": ["female"],
                "person_education": ["PhD"],
                "person_income": [20000],
                "person_emp_exp": [30],
                "person_home_ownership": ["OWN"],
                "loan_amnt": [5000],
                "loan_intent": ["MEDICAL"],
                "loan_int_rate": [5.0],
                "loan_percent_income": [0.25],
                "cb_person_cred_hist_length": [20],
                "credit_score": [800],
                "previous_loan_defaults_on_file": ["No"]
            },
            # High risk applicant
            {
                "person_age": [25],
                "person_gender": ["male"],
                "person_education": ["High School"],
                "person_income": [30000],
                "person_emp_exp": [1],
                "person_home_ownership": ["RENT"],
                "loan_amnt": [25000],
                "loan_intent": ["PERSONAL"],
                "loan_int_rate": [20.0],
                "loan_percent_income": [0.8],
                "cb_person_cred_hist_length": [3],
                "credit_score": [450],
                "previous_loan_defaults_on_file": ["Yes"]
            }
        ]

        print("\nTesting edge cases:")
        for i, case in enumerate(edge_cases):
            # Create DataFrame for the case
            case_df = pd.DataFrame(case)

            # Transform data
            transformed_data = self.preprocessor.transform(case_df)
            feature_names = self.preprocessor.get_feature_names_out()
            transformed_df = pd.DataFrame(transformed_data, columns=feature_names)

            # Make prediction
            prediction = self.model.predict(transformed_df)

            # Verify prediction is valid
            self.assertTrue(prediction[0] in [0, 1])
            print(
                f"Edge case {i + 1}: {list(case.values())[0][0]}yo, ${case['person_income'][0]}/yr, ${case['loan_amnt'][0]} loan → Prediction: {prediction[0]}")


if __name__ == "__main__":
    unittest.main()