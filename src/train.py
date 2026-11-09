from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "train_passenger_survival.csv"
MODEL_DIR = ROOT / "artifacts"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
TARGET = "Survived"
NUMERIC_FEATURES = ["Age", "Fare"]
CATEGORICAL_FEATURES = ["Gender", "SeatClass"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame[FEATURES + [TARGET]].copy()
    for column in NUMERIC_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[TARGET].isna().any() or not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Survived must contain only binary values 0 and 1.")
    if frame[NUMERIC_FEATURES].isna().all().any():
        raise ValueError("At least one numeric feature contains no usable values.")
    if (frame["Age"].dropna() <= 0).any() or (frame["Fare"].dropna() < 0).any():
        raise ValueError("Age must be positive and Fare cannot be negative.")
    if frame.empty or frame[TARGET].nunique() != 2:
        raise ValueError("Training requires non-empty data containing both target classes.")
    return frame


def build_pipeline(estimator) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessing", preprocessing), ("classifier", estimator)])


def train(use_mlflow: bool = True) -> dict:
    data = load_data()
    X, y = data[FEATURES], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42
        ),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        scores = cross_val_score(pipeline, X_train, y_train, cv=folds, scoring="roc_auc")
        cv_results[name] = float(np.mean(scores))
    best_name = max(cv_results, key=cv_results.get)
    best_pipeline = build_pipeline(candidates[best_name])
    best_pipeline.fit(X_train, y_train)
    predictions = best_pipeline.predict(X_test)
    probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    metadata = {
        "model_name": best_name,
        "features": FEATURES,
        "target": TARGET,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset_rows": int(len(data)),
        "cross_validation_roc_auc": cv_results,
        "holdout_metrics": metrics,
        "warning": "Small 70-row dataset; metrics are illustrative and do not establish real-world validity."
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if use_mlflow:
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment("passenger-survival")
        with mlflow.start_run(run_name=best_name):
            mlflow.log_param("selected_model", best_name)
            mlflow.log_param("dataset_rows", len(data))
            mlflow.log_param("random_state", 42)
            mlflow.log_metrics({f"cv_roc_auc_{key}": value for key, value in cv_results.items()})
            mlflow.log_metrics(metrics)
            mlflow.log_dict(metadata, "metadata.json")
            mlflow.sklearn.log_model(best_pipeline, artifact_path="model")
    print(json.dumps(metadata, indent=2))
    print("\nHoldout classification report")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions).tolist())
    print(f"\nSaved model: {MODEL_PATH}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare passenger survival models.")
    parser.add_argument("--no-mlflow", action="store_true", help="Train without logging an MLflow run.")
    args = parser.parse_args()
    train(use_mlflow=not args.no_mlflow)
