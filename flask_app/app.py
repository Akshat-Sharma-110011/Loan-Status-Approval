# At the top of your app.py file, modify the import section:
import os
import sys
import time
import datetime
import traceback
import uuid

# First, ensure the application directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

# Import the PreprocessingPipeline explicitly
from src.data.data_transformation import PreprocessingPipeline

app = Flask(__name__)

# Add health check endpoint for Kubernetes probes
@app.route('/health')
def health_check():
    """Health check endpoint for Kubernetes probes"""
    return jsonify({"status": "healthy"}), 200

# Rest of the file remains the same...

# ===================== MLflow Configuration =====================
DAGSHUB_USERNAME = "Akshat-Sharma-110011"
DAGSHUB_TOKEN = "268f8944c99d48868fa3235eb38ea909e929c70c"
REPO_NAME = "Loan-Status-Approval"
EXPERIMENT_NAME = "CatBoost Optimizer"

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

DEBUG_MODE = True
MODEL_PATH = "./models/model/model.cbm"
PIPELINE_PATH = "./models/preprocessor/preprocessing_pipeline.pkl"

# ===================== Prometheus Metrics =====================
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Prediction counts",
    ["prediction"],
    registry=registry
)


# ===================== Model and Preprocessor Loading =====================
def debug_print(message, level="INFO"):
    if DEBUG_MODE or level in ["ERROR", "CRITICAL"]:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")


try:
    debug_print("Loading preprocessing pipeline...")
    preprocessing_pipeline = joblib.load(PIPELINE_PATH)

    debug_print("Loading CatBoost model...")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    debug_print("Model and pipeline loaded successfully")
    debug_print(f"Model features: {model.feature_names_}")

except Exception as e:
    debug_print(f"Initialization failed: {str(e)}", "CRITICAL")
    traceback.print_exc()
    raise


# ===================== Helper Functions =====================
@app.route("/metrics")
def metrics():
    start_time = time.time()
    REQUEST_COUNT.labels(method="GET", endpoint="/metrics").inc()

    # For Prometheus metrics - this is what Prometheus/Grafana will use
    if request.headers.get('Accept') == CONTENT_TYPE_LATEST:
        response = generate_latest(registry)
        return response, 200, {"Content-Type": CONTENT_TYPE_LATEST}
    else:
        # For HTML dashboard - a simplified version that shows Prometheus metrics
        try:
            # Extract metrics data properly from Prometheus Counters
            request_total = 0
            request_by_endpoint = {}

            # Process request count metrics
            for key, counter in REQUEST_COUNT._metrics.items():
                # The key is typically a tuple of labels like ('GET', '/predict')
                method, endpoint = key
                count_value = counter._value.get()  # Get the actual counter value
                request_total += count_value

                # Group by endpoint for display
                endpoint_label = f"{endpoint}"
                if endpoint_label not in request_by_endpoint:
                    request_by_endpoint[endpoint_label] = 0
                request_by_endpoint[endpoint_label] += count_value

            # Process prediction count metrics
            prediction_total = 0
            prediction_by_type = {}

            for key, counter in PREDICTION_COUNT._metrics.items():
                # The key is a tuple with one element (the prediction type)
                prediction_type = key[0]  # e.g., 'approved' or 'rejected'
                count_value = counter._value.get()
                prediction_total += count_value
                prediction_by_type[prediction_type] = count_value

            # Sort dictionaries for consistent display
            request_by_endpoint = dict(sorted(request_by_endpoint.items()))
            prediction_by_type = dict(sorted(prediction_by_type.items()))

            # Simple metric counts from Prometheus data
            metrics_data = {
                'request_counts': {
                    'total': request_total,
                    'endpoints': request_by_endpoint
                },
                'prediction_counts': {
                    'total': prediction_total,
                    'types': prediction_by_type
                }
            }

            # Include current timestamp for the "Last Updated" display
            current_time = datetime.datetime.now()

            response = render_template('metrics.html', metrics=metrics_data, now=current_time)
        except Exception as e:
            debug_print(f"Error rendering metrics dashboard: {str(e)}", "ERROR")
            debug_print(f"Error trace: {traceback.format_exc()}", "ERROR")
            return render_template('error.html',
                                   error=f"Failed to generate metrics dashboard: {str(e)}",
                                   trace=traceback.format_exc()), 500

    # Record latency for metrics endpoint too
    REQUEST_LATENCY.labels(endpoint="/metrics").observe(time.time() - start_time)

    return response

def preprocess_data(data):
    """
    Preprocess the input data for model prediction.

    Args:
        data (dict): Dictionary containing loan application data

    Returns:
        pandas.DataFrame: Processed data ready for model prediction
    """
    try:
        # Create DataFrame from input data
        input_df = pd.DataFrame([data])

        # Transform using pipeline
        transformed_data = preprocessing_pipeline.transform(input_df)
        features_df = pd.DataFrame(
            transformed_data,
            columns=preprocessing_pipeline.get_feature_names_out()
        )

        debug_print(f"Transformed features:\n{features_df.iloc[0].to_dict()}")
        return features_df
    except Exception as e:
        debug_print(f"Preprocessing error: {str(e)}", "ERROR")
        debug_print(f"Error trace: {traceback.format_exc()}", "ERROR")
        raise


def predict_loan_approval(data):
    """
    Predict loan approval based on input data.

    Args:
        data (dict): Dictionary containing loan application data

    Returns:
        tuple: (status, confidence_percentage)
    """
    try:
        # Validate input data
        for key, value in data.items():
            if value is None or value == '':
                raise ValueError(f"Missing required field: {key}")

        # Process the data and make prediction
        features_df = preprocess_data(data)
        prediction = model.predict(features_df)
        probability = model.predict_proba(features_df)[0][1]

        # For approved loans, relevant probability is the second value (index 1)
        # For rejected loans, relevant probability is the first value (index 0)
        confidence = probability if prediction[0] == 1 else (1 - probability)
        prediction_class = 'Approved' if prediction[0] == 1 else 'Rejected'

        # Record the prediction in metrics
        PREDICTION_COUNT.labels(prediction=prediction_class.lower()).inc()

        debug_print(f"Prediction: {prediction_class} ({confidence:.2%})")

        # Return tuple of status and confidence percentage as string
        return (prediction_class, f"{confidence * 100:.1f}%")

    except Exception as e:
        debug_print(f"Prediction error: {str(e)}", "ERROR")
        debug_print(f"Error trace: {traceback.format_exc()}", "ERROR")
        raise


# ===================== Flask Routes =====================
@app.route('/')
def home():
    start_time = time.time()
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()

    default_values = {
        'person_age': 30,
        'person_gender': 'Male',
        'person_education': 'Bachelor',
        'person_income': 50000,
        'person_emp_exp': 5,
        'person_home_ownership': 'MORTGAGE',
        'loan_amnt': 10000,
        'loan_intent': 'PERSONAL',
        'loan_int_rate': 5.5,
        'loan_percent_income': 20.0,
        'cb_person_cred_hist_length': 10,
        'credit_score': 700,
        'previous_loan_defaults_on_file': 'No'
    }

    response = render_template('index.html', default_values=default_values)

    # Record the request latency
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)

    return response


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()

    try:
        # Define required fields
        required_fields = [
            'person_age', 'person_gender', 'person_education', 'person_income',
            'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
            'credit_score', 'previous_loan_defaults_on_file'
        ]

        # Check for missing fields
        for field in required_fields:
            if field not in request.form:
                debug_print(f"Missing required field: {field}", "ERROR")
                return render_template('error.html',
                                       error=f"Missing required field: {field}"), 400

        try:
            # Extract and convert form data
            form_data = {
                'person_age': int(request.form['person_age']),
                'person_gender': request.form['person_gender'],
                'person_education': request.form['person_education'],
                'person_income': float(request.form['person_income']),
                'person_emp_exp': float(request.form['person_emp_exp']),
                'person_home_ownership': request.form['person_home_ownership'],
                'loan_amnt': float(request.form['loan_amnt']),
                'loan_intent': request.form['loan_intent'],
                'loan_int_rate': float(request.form['loan_int_rate']),
                'loan_percent_income': float(request.form['loan_percent_income']),
                'cb_person_cred_hist_length': int(request.form['cb_person_cred_hist_length']),
                'credit_score': int(request.form['credit_score']),
                'previous_loan_defaults_on_file': request.form['previous_loan_defaults_on_file']
            }
        except ValueError as e:
            debug_print(f"Value conversion error: {str(e)}", "ERROR")
            return render_template('error.html',
                                   error=f"Invalid input data: {str(e)}"), 400

        debug_print(f"Received input data:\n{form_data}")

        # Get prediction result
        status, confidence = predict_loan_approval(form_data)

        response = render_template('result.html',
                                   status=status,
                                   confidence=confidence,
                                   input_data=form_data)

        # Record request latency before returning response
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return response

    except Exception as e:
        error_trace = traceback.format_exc()
        debug_print(f"Prediction error: {str(e)}", "ERROR")
        debug_print(f"Error trace: {error_trace}", "ERROR")

        # Still record latency even for errors
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        # Return a 400 Bad Request for validation errors, 500 for server errors
        if isinstance(e, ValueError):
            return render_template('error.html',
                                   error=str(e),
                                   trace=None), 400
        else:
            return render_template('error.html',
                                   error=str(e),
                                   trace=error_trace), 500

@app.errorhandler(404)
def page_not_found(e):
    start_time = time.time()
    REQUEST_COUNT.labels(method=request.method, endpoint="404").inc()

    response = render_template('error.html', error="Page not found"), 404

    REQUEST_LATENCY.labels(endpoint="404").observe(time.time() - start_time)

    return response


@app.errorhandler(500)
def server_error(e):
    start_time = time.time()
    REQUEST_COUNT.labels(method=request.method, endpoint="500").inc()

    response = render_template('error.html', error="Internal server error"), 500

    REQUEST_LATENCY.labels(endpoint="500").observe(time.time() - start_time)

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)