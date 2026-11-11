"""Tests for Machine Learning and Hybrid Physics-Informed Prognostics."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PRED_DIR = Path(__file__).resolve().parents[1] / "iqac-prediction"
sys.path.insert(0, str(PRED_DIR))

from ml_engine import (
    HybridRULEngine,
    MLRULEngine,
    calculate_nasa_phm_score,
    calculate_prognostic_metrics,
)
from rul_engine import PredictionMethod, RiskLevel


def test_prognostic_metrics():
    """Verify standard prognostic metrics: MAE, RMSE, MAPE, R2, and asymmetric NASA PHM score."""
    y_true = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    # Underestimating prediction (early) and overestimating (late)
    y_pred = np.array([900.0, 2100.0, 2900.0, 4200.0])

    metrics = calculate_prognostic_metrics(y_true, y_pred)
    assert metrics["mae_hours"] == 125.0
    assert metrics["rmse_hours"] > 125.0
    assert metrics["r2_score"] > 0.95
    assert metrics["nasa_phm_score"] > 0.0


def test_ml_engine_refusal_when_uncalibrated():
    """Test Case 14: ML engine refuses to output fake predictions when model is missing or uncalibrated."""
    ml_engine = MLRULEngine(model_bundle_path="non_existent_model.joblib")
    assert not ml_engine.is_trained

    res = ml_engine.predict(avg_temp=55.0, total_spikes_over_85=2, operating_hours=100.0)
    assert res.rul_hours is None
    assert res.health_percentage is None
    assert res.prediction_method == PredictionMethod.ML_BASED.value
    assert any("refuses to output fabricated ML predictions" in lim for lim in res.limitations)


def test_hybrid_engine_fallback_when_ml_missing():
    """Hybrid engine falls back gracefully to physics-based bounds when ML is not trained."""
    hybrid = HybridRULEngine(model_bundle_path="non_existent_model.joblib")
    res = hybrid.evaluate(avg_temp=50.0, total_spikes_over_85=1, operating_hours=200.0, reading_count=30)

    assert res.rul_hours is not None
    assert res.prediction_method == PredictionMethod.PHYSICS_BASED.value
    assert any("operating in calibrated Physics mode" in lim for lim in res.limitations)


def test_hybrid_engine_fusion_and_disagreement():
    """Hybrid engine combines both methods, detects model disagreement, and widens uncertainty."""
    hybrid = HybridRULEngine()  # Loads existing calibrated bundle
    res = hybrid.evaluate(avg_temp=65.0, total_spikes_over_85=5, operating_hours=300.0, reading_count=40)

    assert res.rul_hours is not None
    assert res.confidence_interval is not None
    assert res.confidence_interval["lower_hours"] <= res.rul_hours <= res.confidence_interval["upper_hours"]
    assert "metadata" in res.to_dict()


def test_invalid_temp_in_ml_engine():
    """Invalid temperature input (<= 0°C or > 125°C) aborts ML inference cleanly."""
    ml_engine = MLRULEngine()
    for bad_temp in [None, 0.0, -10.0, 140.0]:
        res = ml_engine.predict(avg_temp=bad_temp, total_spikes_over_85=0, operating_hours=50.0)
        assert res.rul_hours is None
        assert res.risk_level == RiskLevel.UNKNOWN.value

