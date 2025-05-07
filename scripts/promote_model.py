import os
import mlflow


def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "Akshat-Sharma-110011"
    repo_name = "Loan-Status-Approval"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()
    model_name = "Catboost_model"

    # Check if there are any models in staging
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

    if not staging_versions:
        print(f"No models found in Staging for {model_name}. Checking for registered models...")
        # Get all available versions to promote the latest one
        all_versions = client.get_latest_versions(model_name)

        if not all_versions:
            raise ValueError(f"No versions found for model {model_name}")

        # Sort versions by creation timestamp and get the latest one
        latest_version = sorted(all_versions, key=lambda x: x.creation_timestamp, reverse=True)[0]
        version_to_promote = latest_version.version
        print(f"Found latest model version: {version_to_promote}")
    else:
        # Get the latest version in staging
        version_to_promote = staging_versions[0].version

    # Archive the current production model if any exists
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        print(f"Archiving current production model version {version.version}")
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    print(f"Promoting model version {version_to_promote} to Production")
    client.transition_model_version_stage(
        name=model_name,
        version=version_to_promote,
        stage="Production"
    )
    print(f"Model version {version_to_promote} successfully promoted to Production")


if __name__ == "__main__":
    promote_model()