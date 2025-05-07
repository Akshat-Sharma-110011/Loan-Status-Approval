from flask import Flask, render_template, request
import joblib
import pandas as pd
import traceback
from catboost import CatBoostClassifier
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the PreprocessingPipeline class to fix unpickling
from src.data.data_transformation import PreprocessingPipeline

app = Flask(__name__)

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

        debug_print(f"Received input data:\n{form_data}")

        # Create DataFrame
        input_df = pd.DataFrame([form_data])

        # Transform using pipeline
        transformed_data = preprocessing_pipeline.transform(input_df)
        features_df = pd.DataFrame(
            transformed_data,
            columns=preprocessing_pipeline.get_feature_names_out()
        )

        debug_print(f"Transformed features:\n{features_df.iloc[0].to_dict()}")

        # Make prediction
        prediction = model.predict(features_df)
        probability = model.predict_proba(features_df)[0][1]
        prediction_class = 'approved' if prediction[0] == 1 else 'rejected'

        PREDICTION_COUNT.labels(prediction=prediction_class).inc()

        debug_print(f"Prediction: {prediction_class} ({probability:.2%})")

        response = render_template('result.html',
                                   status='Approved' if prediction[0] == 1 else 'Rejected',
                                   confidence=f"{probability * 100:.1f}%",
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

        return render_template('error.html',
                               error=str(e),
                               trace=error_trace), 500


@app.route("/metrics")
def metrics():
    start_time = time.time()
    REQUEST_COUNT.labels(method="GET", endpoint="/metrics").inc()

    response = generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    # Record latency for metrics endpoint too
    REQUEST_LATENCY.labels(endpoint="/metrics").observe(time.time() - start_time)

    return response


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