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


def train_and_log_model(save_local: bool = True) -> None:

    # Detect if running on Databricks
    running_on_databricks = os.environ.get("DATABRICKS_RUNTIME") is not None

    # Set MLflow registry depending on environment
    if running_on_databricks:
        mlflow.set_registry_uri("databricks")  

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(model, artifact_path="model")

        if save_local:
            artifacts_dir = pathlib.Path("artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            joblib.dump(model, artifacts_dir / "model.joblib")

        print(f"Logged model with ROC AUC={auc:.4f}")


if __name__ == "__main__":
    mlflow.set_experiment("fraud-mlops-demo-local")
    train_and_log_model()