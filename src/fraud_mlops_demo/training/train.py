from __future__ import annotations

import pathlib
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def train_and_log_model(save_local: bool = True):

    # Detect if running on Databricks workspace
    on_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ

    if on_databricks:
        # Use Databricks workspace MLflow
        mlflow.set_registry_uri("databricks")
        experiment_name = "/Shared/fraud-mlops-demo"  # absolute workspace path
    else:
        # Local / Jupyter / VS Code
        local_registry = pathlib.Path.home() / "mlflow_registry"
        local_registry.mkdir(exist_ok=True)
        mlflow.set_registry_uri(f"file://{local_registry}")
        experiment_name = "fraud-mlops-demo-local"  # simple local experiment

    mlflow.set_experiment(experiment_name)

    # Load dataset
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Start MLflow run
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        # Log parameters and metrics
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_metric("roc_auc", auc)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Optionally save locally
        if save_local:
            artifacts_dir = pathlib.Path("artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            joblib.dump(model, artifacts_dir / "model.joblib")

        print(f"Logged model with ROC AUC={auc:.4f}")


if __name__ == "__main__":
    train_and_log_model()