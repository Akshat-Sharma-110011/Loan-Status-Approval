import unittest
from unittest.mock import patch, MagicMock
from flask_app.app import app


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

    @patch('flask_app.app.predict_loan_approval')
    def test_predict_page_approve(self, mock_predict):
        """Test if the prediction page works for approved loans"""
        # Mock the prediction function to return approval
        mock_predict.return_value = ('Approved', '92.5%')

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
        self.assertIn(b'Loan Application Approved', response.data)
        self.assertIn(b'Confidence: 92.5%', response.data)
        self.assertIn(b'Congratulations!', response.data)

    @patch('flask_app.app.predict_loan_approval')
    def test_predict_page_reject(self, mock_predict):
        """Test if the prediction page works for rejected loans"""
        # Mock the prediction function to return rejection
        mock_predict.return_value = ('Rejected', '85.3%')

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
        self.assertIn(b'Loan Application Rejected', response.data)
        self.assertIn(b'Confidence: 85.3%', response.data)
        self.assertIn(b'Tips to Improve Approval Chances', response.data)

    def test_missing_field_validation(self):
        """Test validation for missing required fields"""
        # Submit form with missing required fields
        form_data = {
            'person_age': 35,
            # Missing most fields
            'loan_amnt': 15000.0
        }

        response = self.client.post('/predict', data=form_data)
        # Should either get a 400 Bad Request, or show the form again with error messages
        self.assertNotEqual(response.status_code, 500)  # At least make sure it doesn't cause a server error

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

        response = self.client.post('/predict', data=form_data)
        # Should either get a 400 Bad Request, redirect to an error page, or show the form again with error messages
        self.assertNotEqual(response.status_code, 500)  # At least make sure it doesn't cause a server error

    @patch('flask_app.app.get_metrics_data')
    def test_metrics_page(self, mock_metrics):
        """Test if the metrics page loads correctly"""
        # Mock the metrics data function
        mock_metrics.return_value = {
            'request_count': 152,
            'get_requests': 89,
            'post_requests': 63,
            'predictions': 63,
            'approved_percent': 71,
            'rejected_percent': 29,
            'avg_response_time': 218
        }

        response = self.client.get('/metrics')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Live Metrics', response.data)

    def test_debug_console(self):
        """Test if the debug console page loads correctly"""
        response = self.client.get('/debug')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Debug Console', response.data)
        self.assertIn(b'Debugging Tools', response.data)
        self.assertIn(b'Log Console', response.data)

    def test_error_page(self):
        """Test if the error page is shown correctly"""
        # Assuming there's a route that simulates an error for testing
        with patch('flask_app.app.handle_error', side_effect=Exception('Test error')):
            response = self.client.get('/simulate-error')
            self.assertIn(b'Oops! Something went wrong', response.data)
            self.assertIn(b'Test error', response.data)

    def test_404_error_handler(self):
        """Test if 404 errors are handled correctly"""
        response = self.client.get('/non-existent-page')
        self.assertEqual(response.status_code, 404)
        # Check if it's using your custom error page
        self.assertIn(b'Oops! Something went wrong', response.data)

    def test_500_error_handler(self):
        """Test if 500 errors are handled correctly"""
        # This might require a special route that deliberately raises an exception
        with patch('flask_app.app.some_function', side_effect=Exception('Test server error')):
            response = self.client.get('/route-with-server-error')
            self.assertEqual(response.status_code, 500)
            self.assertIn(b'Oops! Something went wrong', response.data)


if __name__ == '__main__':
    unittest.main()