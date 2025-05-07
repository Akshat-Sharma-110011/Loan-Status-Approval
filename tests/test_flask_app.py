import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# First, make sure we add the parent directory to sys.path
# This ensures imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the PreprocessingPipeline class BEFORE importing the Flask app
# This ensures the class is in the global namespace when joblib tries to unpickle
from src.data.data_transformation import PreprocessingPipeline

# Create mocks for the model and pipeline to avoid loading actual files during testing
mock_pipeline = MagicMock()
mock_pipeline.transform.return_value = [[1, 0, 0, 1, 0]]  # Example transformed data
mock_pipeline.get_feature_names_out.return_value = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

mock_model = MagicMock()
mock_model.predict.return_value = [1]  # Example prediction (approved)
mock_model.predict_proba.return_value = [[0.25, 0.75]]  # Example probability
mock_model.feature_names_ = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

# Use patch to replace joblib.load and model.load_model before importing the app
with patch('joblib.load', return_value=mock_pipeline), \
        patch('catboost.CatBoostClassifier.load_model', return_value=None):
    # Now import the app after setting up the environment and patches
    from flask_app.app import app


# The tests can now run without actually loading the model or pipeline files
class LoanApprovalPredictorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing

    def test_home_page(self):
        """Test if the home page loads correctly"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Loan Approval Predictor</title>', response.data)
        self.assertIn(b'Check Your Loan Approval Chances', response.data)
        self.assertIn(b'Loan Application Form', response.data)

    def test_about_section(self):
        """Test if the about section is present on the home page"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="about"', response.data)
        self.assertIn(b'About This Predictor', response.data)
        self.assertIn(b'AI-Powered', response.data)

    def test_contact_section(self):
        """Test if the contact section is present on the home page"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="contact"', response.data)
        self.assertIn(b'Contact Us', response.data)
        self.assertIn(b'If you have any questions or feedback', response.data)

    def test_metrics_page_link(self):
        """Test if the metrics page link is present"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'href="/metrics"', response.data)

    # Update: Using direct mock patches on the app functions instead of non-existent function
    @patch('flask_app.app.model.predict')
    @patch('flask_app.app.model.predict_proba')
    def test_predict_page_approve(self, mock_predict_proba, mock_predict):
        """Test if the prediction page works for approved loans"""
        # Set up the mocks for model predictions
        mock_predict.return_value = [1]  # Approved
        mock_predict_proba.return_value = [[0.075, 0.925]]  # 92.5% confidence

        form_data = {
            'person_age': 35,
            'person_gender': 'Male',
            'person_education': 'Bachelor',
            'person_income': 65000.0,
            'person_emp_exp': 8.0,
            'person_home_ownership': 'MORTGAGE',
            'loan_amnt': 15000.0,
            'loan_intent': 'PERSONAL',
            'loan_int_rate': 6.5,
            'loan_percent_income': 23.0,
            'cb_person_cred_hist_length': 12,
            'credit_score': 720,
            'previous_loan_defaults_on_file': 'No'
        }

        response = self.client.post('/predict', data=form_data)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Approved', response.data)
        self.assertIn(b'92.5%', response.data)

    # Update: Using direct mock patches on the app functions instead of non-existent function
    @patch('flask_app.app.model.predict')
    @patch('flask_app.app.model.predict_proba')
    def test_predict_page_reject(self, mock_predict_proba, mock_predict):
        """Test if the prediction page works for rejected loans"""
        # Set up the mocks for model predictions
        mock_predict.return_value = [0]  # Rejected
        mock_predict_proba.return_value = [[0.853, 0.147]]  # 85.3% confidence for rejection

        form_data = {
            'person_age': 25,
            'person_gender': 'Female',
            'person_education': 'High School',
            'person_income': 30000.0,
            'person_emp_exp': 1.5,
            'person_home_ownership': 'RENT',
            'loan_amnt': 25000.0,
            'loan_intent': 'DEBT_CONSOLIDATION',
            'loan_int_rate': 12.5,
            'loan_percent_income': 83.0,
            'cb_person_cred_hist_length': 3,
            'credit_score': 580,
            'previous_loan_defaults_on_file': 'Yes'
        }

        response = self.client.post('/predict', data=form_data)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Rejected', response.data)
        self.assertIn(b'85.3%', response.data)

    def test_missing_field_validation(self):
        """Test validation for missing required fields"""
        # Submit form with missing required fields
        form_data = {
            'person_age': 35,
            # Missing most fields
            'loan_amnt': 15000.0
        }

        # Update expectation: Since this is handled gracefully with a 400 response in app.py
        response = self.client.post('/predict', data=form_data)
        self.assertEqual(response.status_code, 400)  # Expect a 400 Bad Request

    def test_invalid_data_validation(self):
        """Test validation for invalid data types"""
        # Submit form with invalid data types
        form_data = {
            'person_age': 'invalid',  # String instead of integer
            'person_gender': 'Male',
            'person_education': 'Bachelor',
            'person_income': 'invalid',  # String instead of float
            'person_emp_exp': 8.0,
            'person_home_ownership': 'MORTGAGE',
            'loan_amnt': 15000.0,
            'loan_intent': 'PERSONAL',
            'loan_int_rate': 6.5,
            'loan_percent_income': 23.0,
            'cb_person_cred_hist_length': 12,
            'credit_score': 720,
            'previous_loan_defaults_on_file': 'No'
        }

        # Update expectation: Since this is handled gracefully with a 400 response in app.py
        response = self.client.post('/predict', data=form_data)
        self.assertEqual(response.status_code, 400)  # Expect a 400 Bad Request

    @patch('flask_app.app.get_metrics_data')
    @patch('flask_app.app.render_template', return_value='Mocked metrics')
    def test_metrics_page(self, mock_render_template, mock_get_metrics_data):
        """Test if the metrics page loads correctly"""
        # Mock get_metrics_data to return dummy metrics
        mock_get_metrics_data.return_value = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.79,
            'f1_score': 0.80,
            'confusion_matrix': [[150, 30], [25, 200]],
            'roc_auc': 0.89
        }

        # Create a mock for the Prometheus generate_latest function
        with patch('flask_app.app.generate_latest',
                   return_value=b'app_request_count{method="GET"} 1.0'):
            # Test Prometheus metrics endpoint
            prometheus_headers = {'Accept': 'application/openmetrics-text'}
            response = self.client.get('/metrics', headers=prometheus_headers)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'app_request_count', response.data)

            # Test HTML metrics page
            standard_headers = {}
            response = self.client.get('/metrics', headers=standard_headers)
            self.assertEqual(response.status_code, 200)
            # Since we've mocked render_template, we only need to check the response is as expected
            self.assertEqual(response.data, b'Mocked metrics')

            # Verify render_template was called correctly
            mock_render_template.assert_called_with('metrics.html', metrics=mock_get_metrics_data.return_value)

    def test_404_error_handler(self):
        """Test if 404 errors are handled correctly"""
        response = self.client.get('/non-existent-page')
        self.assertEqual(response.status_code, 404)
        # Check if it's using your custom error page
        self.assertIn(b'error', response.data)  # Adjust based on your actual error page content


if __name__ == '__main__':
    unittest.main()