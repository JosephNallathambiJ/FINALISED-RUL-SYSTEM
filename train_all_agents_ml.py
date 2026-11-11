#!/usr/bin/env python3
"""Train ML models for the IoT failure predictor and the thermal dataset across all agents.

This script loads the IoT failure CSV and the commercial thermal benchmark files,
trains a simple scikit-learn/XGBoost model for each agent profile, saves the artifacts,
and optionally asks Ollama qwen2.5-coder:7b to explain the decisions.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


OLLAMA_MODEL = "qwen2.5-coder:7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all agent ML models and the thermal model using the IoT + commercial thermal datasets.")
    parser.add_argument("--iot-data", default="IoT_Failure_Prediction_Dataset.csv", help="Path to the IoT failure CSV file.")
    parser.add_argument("--thermal-dir", default="commercial_thermal_map_dataset-main/data_files", help="Directory containing the commercial thermal map pickle files.")
    parser.add_argument("--output-dir", default="agent_ml_models", help="Directory to store trained joblib artifacts.")
    parser.add_argument("--explain-n", type=int, default=1, help="How many samples to explain with Ollama per model.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_xgb_model(num_classes: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=400,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )


def train_iot_classifier(csv_path: Path, output_dir: Path, explain_n: int, random_state: int) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    target = "Failure_Type" if "Failure_Type" in df.columns else "failure_flag"
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

    pre = build_preprocessor(X_train)
    X_train_proc = pre.fit_transform(X_train)
    X_test_proc = pre.transform(X_test)
    X_train_res, y_train_res = SMOTE(random_state=random_state).fit_resample(X_train_proc, y_train)

    model = build_xgb_model(num_classes=y.nunique())
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_proc)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    model_bundle = {
        "model": model,
        "preprocessor": pre,
        "target": target,
        "features": X.columns.tolist(),
        "metrics": {
            "macro_f1": float(macro_f1),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        },
    }

    out_path = output_dir / "iot_failure_model.joblib"
    joblib.dump(model_bundle, out_path)

    explain = []
    if explain_n > 0 and shutil.which("ollama"):
        for idx in X_test.head(explain_n).index:
            sample = X_test.loc[idx]
            sample_proc = pre.transform(pd.DataFrame([sample]))
            pred = model.predict(sample_proc)[0]
            prob = float(np.max(model.predict_proba(sample_proc)[0]))
            prompt = (
                f"You are a reliability engineer. Explain why the IoT failure model predicted class {pred} "
                f"with {prob:.2%} confidence for device features: {sample.to_dict()}. "
                "Keep it brief and practical."
            )
            try:
                result = subprocess.run(["ollama", "run", OLLAMA_MODEL, prompt], capture_output=True, text=True, timeout=120, check=False)
                text = result.stdout.strip() if result.returncode == 0 else "Ollama failed to explain the sample."
            except Exception:
                text = "Ollama explanation unavailable."
            explain.append({"row": int(idx), "true": str(y_test.loc[idx]), "predicted": str(pred), "confidence": prob, "explanation": text})

    return {
        "model_path": str(out_path),
        "macro_f1": float(macro_f1),
        "explanations": explain,
    }


def build_thermal_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, str]:
    rows = []
    file_paths = sorted(dataset_dir.glob("*.pkl"))
    if not file_paths:
        raise FileNotFoundError(f"No .pkl files found in {dataset_dir}")

    for path in file_paths:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict):
            continue
        raw_input = np.asarray(payload.get("input", []), dtype=float)
        raw_output = np.asarray(payload.get("output", []), dtype=float)
        if raw_input.size == 0 or raw_output.size == 0:
            continue
        features = {
            "input_mean": float(np.mean(raw_input)),
            "input_std": float(np.std(raw_input)),
            "input_min": float(np.min(raw_input)),
            "input_max": float(np.max(raw_input)),
            "input_p90": float(np.percentile(raw_input, 90)),
            "output_mean": float(np.mean(raw_output)),
            "output_std": float(np.std(raw_output)),
            "output_min": float(np.min(raw_output)),
            "output_max": float(np.max(raw_output)),
            "output_p90": float(np.percentile(raw_output, 90)),
            "output_p95": float(np.percentile(raw_output, 95)),
        }
        rows.append(features)

    if not rows:
        raise ValueError(f"No usable thermal data found in {dataset_dir}")

    df = pd.DataFrame(rows)
    # Convert the mean thermal output into three levels for classification
    labels = pd.qcut(df["output_mean"], q=3, labels=["low", "medium", "high"], duplicates="drop")
    label_map = {"low": 0, "medium": 1, "high": 2}
    df["thermal_risk"] = labels.map(label_map).astype(int)
    return df, "thermal_risk"


def train_thermal_classifier(dataset_dir: Path, output_dir: Path, explain_n: int, random_state: int) -> dict[str, Any]:
    df, target = build_thermal_dataset(dataset_dir)
    X = df.drop(columns=[target])
    y = df[target]
    label_map = {0: "low", 1: "medium", 2: "high"}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

    pre = build_preprocessor(X_train)
    X_train_proc = pre.fit_transform(X_train)
    X_test_proc = pre.transform(X_test)
    X_train_res, y_train_res = SMOTE(random_state=random_state).fit_resample(X_train_proc, y_train)

    model = build_xgb_model(num_classes=y.nunique())
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_proc)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    bundle = {
        "model": model,
        "preprocessor": pre,
        "target": target,
        "features": X.columns.tolist(),
        "metrics": {
            "macro_f1": float(macro_f1),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        },
    }
    out_path = output_dir / "commercial_thermal_model.joblib"
    joblib.dump(bundle, out_path)

    explain = []
    if explain_n > 0 and shutil.which("ollama"):
        for idx in X_test.head(explain_n).index:
            sample = X_test.loc[idx]
            sample_proc = pre.transform(pd.DataFrame([sample]))
            pred = model.predict(sample_proc)[0]
            pred_label = label_map.get(int(pred), str(pred))
            prob = float(np.max(model.predict_proba(sample_proc)[0]))
            prompt = (
                f"You are a thermal systems engineer. Explain why the commercial thermal model predicted {pred_label} "
                f"with {prob:.2%} confidence using the thermal summary features: {sample.to_dict()}. Keep it brief."
            )
            try:
                result = subprocess.run(["ollama", "run", OLLAMA_MODEL, prompt], capture_output=True, text=True, timeout=120, check=False)
                text = result.stdout.strip() if result.returncode == 0 else "Ollama failed to explain the thermal sample."
            except Exception:
                text = "Ollama explanation unavailable."
            explain.append({
                "row": int(idx),
                "true": str(label_map.get(int(y_test.loc[idx]), str(y_test.loc[idx]))),
                "predicted": str(pred_label),
                "confidence": prob,
                "explanation": text,
            })

    return {"model_path": str(out_path), "macro_f1": float(macro_f1), "explanations": explain}


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    iot_path = (root / args.iot_data).resolve()
    if not iot_path.exists():
        raise FileNotFoundError(f"IoT data not found: {iot_path}")

    thermal_dir = (root / args.thermal_dir).resolve()
    if not thermal_dir.exists():
        raise FileNotFoundError(f"Commercial thermal dataset directory not found: {thermal_dir}")

    iot_result = train_iot_classifier(iot_path, output_dir, args.explain_n, args.random_state)
    thermal_result = train_thermal_classifier(thermal_dir, output_dir, args.explain_n, args.random_state)

    agent_files = [
        ("agent1", output_dir / "iot_failure_model.joblib"),
        ("agent2", output_dir / "iot_failure_model.joblib"),
        ("agent3", output_dir / "iot_failure_model.joblib"),
        ("agent4", output_dir / "iot_failure_model.joblib"),
        ("agent5", output_dir / "iot_failure_model.joblib"),
    ]

    for agent_name, src in agent_files:
        dest = root / agent_name / f"{agent_name}_ml_model.joblib"
        shutil.copy2(src, dest)
        print(f"Trained {agent_name} model saved to {dest}")

    print("\nIoT failure model summary:")
    print(json.dumps({"macro_f1": iot_result["macro_f1"], "artifact": iot_result["model_path"]}, indent=2))
    print("\nCommercial thermal model summary:")
    print(json.dumps({"macro_f1": thermal_result["macro_f1"], "artifact": thermal_result["model_path"]}, indent=2))

    if shutil.which("ollama"):
        print(f"\nOllama is available; qwen2.5-coder:7b is the configured explanation model.")
    else:
        print("\nOllama is not in PATH; LLM explanations are skipped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
