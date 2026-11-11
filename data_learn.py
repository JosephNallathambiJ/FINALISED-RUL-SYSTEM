#!/usr/bin/env python3
"""Train an IoT failure-prediction model with class imbalance handling and LLM explanations.

Example:
    python3 data_learn.py \
      --data IoT_Failure_Prediction_Dataset.csv \
      --target Failure_Type \
      --model-output iot_failure_model.joblib \
      --explain-n 5
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


MODEL_NAME = "qwen2.5-coder:7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an IoT failure prediction model while handling class imbalance and explaining results with Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        help="Path to the CSV dataset. Defaults to the first matching dataset in the current folder.",
    )
    parser.add_argument("--target", default="Failure_Type", help="Target column name, such as Failure_Type or failure_flag.")
    parser.add_argument("--model-output", default="iot_failure_model.joblib", help="Destination path for the trained model artifact.")
    parser.add_argument("--explain-n", type=int, default=5, help="Number of validation samples to explain via Ollama. Use 0 to disable.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out validation ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used for split and model reproducibility.")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-driven explanations even if Ollama is present.")
    return parser.parse_args()


def resolve_target_column(df: pd.DataFrame, target_name: str) -> str:
    """Resolve a target column name against common aliases and DataFrame columns."""
    if target_name in df.columns:
        return target_name

    normalized = {col.lower(): col for col in df.columns}
    if target_name.lower() in normalized:
        return normalized[target_name.lower()]

    aliases = {
        "failure_flag": ["failure_flag", "failure_type", "failureType", "label"],
        "failure_type": ["failure_type", "failure_flag", "failureType", "label"],
    }

    for alternatives in aliases.values():
        for alt in alternatives:
            if alt.lower() == target_name.lower():
                for candidate in alternatives:
                    if candidate.lower() in normalized:
                        return normalized[candidate.lower()]

    raise ValueError(f"Target column '{target_name}' not found. Available columns: {list(df.columns)}")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a feature preprocessing stage for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_sklearn_baseline(n_classes: int) -> LogisticRegression:
    """Return a balanced sklearn logistic-regression baseline for comparison."""
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        n_jobs=None,
    )


def build_xgboost_model(n_classes: int) -> XGBClassifier:
    """Return an XGBoost model tuned for multiclass failure prediction and class imbalance."""
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )


def build_resampling_pipeline(X: pd.DataFrame, random_state: int = 42) -> ImbPipeline:
    """Build the training-only preprocessing and SMOTE pipeline."""
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("smote", SMOTE(random_state=random_state, k_neighbors=5)),
        ]
    )


def explain_sample_with_ollama(sample: pd.Series, prediction: Any, probability: float, feature_names: list[str], model_name: str = MODEL_NAME) -> str:
    """Ask a local Ollama model to explain a single prediction."""
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        return "Ollama is not installed or not on PATH; LLM explanation skipped."

    feature_summary = []
    for name in feature_names:
        value = sample.get(name, None)
        if value is None:
            continue
        feature_summary.append(f"{name}={value}")
    feature_text = ", ".join(feature_summary) if feature_summary else "No feature values available"

    prompt = f"""
You are a senior reliability engineer. Explain this IoT device failure prediction in simple business-friendly English.

- Predicted class: {prediction}
- Confidence: {probability:.2%}
- Device features: {feature_text}
- Explain which signals matter most, identify the likely root cause, and suggest the best next maintenance action.
Keep the answer concise, but useful. Do not overstate certainty.
""".strip()

    try:
        proc = subprocess.run(["ollama", "run", model_name, prompt], capture_output=True, text=True, timeout=120, check=False)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            return f"LLM explanation unavailable: {stderr or 'the model failed to respond.'}"
        return proc.stdout.strip() or "LLM returned no explanation."
    except FileNotFoundError:
        return "Ollama CLI not found; LLM explanation skipped."
    except subprocess.TimeoutExpired:
        return "LLM explanation timed out; retry later or ensure the model is available."


def evaluate_with_cv(X: pd.DataFrame, y: pd.Series, random_state: int) -> tuple[float, list[float]]:
    """Run stratified cross-validation using macro-F1 as the validation metric."""
    cv = StratifiedKFold(n_splits=min(5, y.nunique()), shuffle=True, random_state=random_state)
    scores: list[float] = []

    for train_idx, val_idx in cv.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        resample_pipeline = build_resampling_pipeline(X_train_fold, random_state=random_state)
        X_train_res, y_train_res = resample_pipeline.fit_resample(X_train_fold, y_train_fold)
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_res)

        model = build_xgboost_model(n_classes=y.nunique())
        model.fit(X_train_res, y_train_res, sample_weight=sample_weights)
        y_pred_fold = model.predict(resample_pipeline.named_steps["preprocessor"].transform(X_val_fold))
        scores.append(f1_score(y_val_fold, y_pred_fold, average="macro", zero_division=0))

    mean_score = float(np.mean(scores))
    return mean_score, scores


def print_feature_importance(model, feature_names: list[str]) -> None:
    """Display a sorted feature-importance summary."""
    if not hasattr(model, "feature_importances_"):
        print("Feature importance is not available for this model type.")
        return

    importances = model.feature_importances_
    feature_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    feature_df = feature_df.sort_values("importance", ascending=False).reset_index(drop=True)

    print("\nTop feature importances:")
    print(feature_df.head(15).to_string(index=False, formatters={"importance": lambda v: f"{v:.4f}"}))


def auto_detect_data_path(provided_path: str | None) -> Path:
    """Return the user-provided dataset or detect a default dataset in the current directory."""
    if provided_path:
        return Path(provided_path)

    for name in ["IoT_Failure_Prediction_Dataset.csv", "iot_failure_prediction_dataset.csv", "data.csv"]:
        candidate = Path.cwd() / name
        if candidate.exists():
            return candidate

    for path in sorted(Path.cwd().glob("**/*.csv"), key=lambda p: p.name.lower()):
        if any(token in path.name.lower() for token in ["failure", "iot", "telemetry"]):
            return path

    raise FileNotFoundError("No dataset file was found in the current directory. Pass --data /path/to/your_dataset.csv explicitly.")


def dump_training_summary(model, y_true, y_pred, feature_names: list[str], target_name: str, model_output: str, explain_samples: list[dict[str, Any]] | None) -> None:
    """Persist the model with metadata for downstream inference and explanation."""
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    payload = {
        "model": model,
        "target": target_name,
        "feature_names": feature_names,
        "metrics": metrics,
        "explain_samples": explain_samples or [],
    }
    joblib.dump(payload, model_output)


def main() -> int:
    args = parse_args()
    data_path = auto_detect_data_path(args.data)
    model_output = Path(args.model_output)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {data_path}")

    target_column = resolve_target_column(df, args.target)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing from dataset.")

    features = [col for col in df.columns if col != target_column]
    if not features:
        raise ValueError("No feature columns were found. The dataset may only contain the target column.")

    X = df[features]
    y = df[target_column]

    cv_macro_f1, cv_scores = evaluate_with_cv(X, y, args.random_state)
    print(f"\nStratified CV Macro F1: {cv_macro_f1:.4f}")
    print(f"Per-fold Macro F1 scores: {[round(score, 4) for score in cv_scores]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    resample_pipeline = build_resampling_pipeline(X_train, random_state=args.random_state)
    X_train_res, y_train_res = resample_pipeline.fit_resample(X_train, y_train)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_res)

    xgb_model = build_xgboost_model(n_classes=y.nunique())
    xgb_model.fit(X_train_res, y_train_res, sample_weight=sample_weights)

    X_test_proc = resample_pipeline.named_steps["preprocessor"].transform(X_test)
    y_pred_xgb = xgb_model.predict(X_test_proc)
    holdout_macro_f1 = f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)

    sklearn_model = build_sklearn_baseline(n_classes=y.nunique())
    sklearn_model.fit(X_train_res, y_train_res)
    y_pred_sklearn = sklearn_model.predict(resample_pipeline.named_steps["preprocessor"].transform(X_test))

    print(f"Training rows: {len(df)}")
    print(f"Feature columns: {len(features)}")
    print(f"Target column: {target_column}")
    print(f"XGBoost Hold-out Macro F1: {holdout_macro_f1:.4f}")
    print(f"Sklearn Baseline Hold-out Macro F1: {f1_score(y_test, y_pred_sklearn, average='macro', zero_division=0):.4f}")
    print(f"\nXGBoost classification report:\n{classification_report(y_test, y_pred_xgb, zero_division=0)}")
    print(f"\nSklearn baseline classification report:\n{classification_report(y_test, y_pred_sklearn, zero_division=0)}")

    feature_names = resample_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    print_feature_importance(xgb_model, feature_names)

    model_output.parent.mkdir(parents=True, exist_ok=True)

    explain_samples: list[dict[str, Any]] = []
    if args.explain_n > 0 and not args.no_llm:
        sample_indices = list(X_test.index[: min(args.explain_n, len(X_test))])
        for idx in sample_indices:
            sample = X_test.loc[idx]
            sample_proc = resample_pipeline.named_steps["preprocessor"].transform(pd.DataFrame([sample]))
            pred = xgb_model.predict(sample_proc)[0]
            proba = xgb_model.predict_proba(sample_proc)[0]
            conf = float(np.max(proba))
            explanation = explain_sample_with_ollama(sample, pred, conf, features)
            explain_samples.append(
                {
                    "row_index": int(idx),
                    "true_label": str(y_test.loc[idx]),
                    "predicted_label": str(pred),
                    "confidence": conf,
                    "explanation": explanation,
                }
            )

        print("\nLLM explanations:")
        for item in explain_samples:
            print(f"\nSample row {item['row_index']} -> {item['predicted_label']} (confidence={item['confidence']:.2%})")
            print(item["explanation"])

    dump_training_summary(xgb_model, y_test, y_pred_xgb, features, target_column, str(model_output), explain_samples)
    print(f"\nSaved trained model to: {model_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
