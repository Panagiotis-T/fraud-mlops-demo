from __future__ import annotations

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def train_and_log_model() -> None:
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

        print(f"Logged model with ROC AUC={auc:.4f}")


if __name__ == "__main__":
    train_and_log_model()