import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import os
import shutil
import pandas as pd
import numpy as np
import pytest
from app.ai.model import PandemicModel  # Remplace par le vrai nom du fichier si différent

@pytest.fixture
def sample_df():
    dates = pd.date_range("2023-01-01", periods=60)
    data = {
        f"lag_{i}": np.random.rand(60) for i in range(1, 31)  # ← ici, 30 lags
    }
    data.update({
        "rolling_7_mean": np.random.rand(60),
        "rolling_30_mean": np.random.rand(60),
        "day_of_week": np.random.randint(0, 7, 60),
        "day_of_month": np.random.randint(1, 29, 60),
        "month": np.random.randint(1, 13, 60),
        "new_cases": np.random.randint(0, 1000, 60),
        "new_deaths": np.random.randint(0, 100, 60),
        "new_recovered": np.random.randint(0, 200, 60),
    })
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def model_dir(tmp_path):
    """Répertoire temporaire pour les modèles."""
    return tmp_path / "models"

@pytest.fixture
def model(model_dir):
    return PandemicModel(model_dir=str(model_dir))

def test_train_model_and_metrics(model, sample_df):
    features = [f"lag_{i}" for i in range(1, 31)] + ["rolling_7_mean", "rolling_30_mean", "day_of_week", "day_of_month", "month"]
    trained_model, metrics = model.train_model(sample_df, "new_cases", features)
    assert trained_model is not None
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics

def test_save_and_load_model(model, sample_df):
    features = ["lag_1", "lag_2", "rolling_7_mean", "rolling_30_mean", "day_of_week", "day_of_month", "month"]
    trained_model, _ = model.train_model(sample_df, "new_cases", features)
    model.save_model(trained_model, "new_cases")

    loaded_model = model.load_model("new_cases")
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict")

def test_predict_future(model, sample_df):
    features = ["lag_1", "lag_2", "rolling_7_mean", "rolling_30_mean", "day_of_week", "day_of_month", "month"]
    model.train_model(sample_df, "new_cases", features)
    model.save_model(model.load_model("new_cases"), "new_cases")
    preds = model.predict_future(sample_df, "new_cases", features, days_ahead=5)
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == 5
    assert "predicted_new_cases" in preds.columns

def test_predict_multiple_targets(model, sample_df):
    sample_df["new_deaths"] = np.random.randint(0, 100, len(sample_df))
    sample_df["new_recovered"] = np.random.randint(0, 200, len(sample_df))
    features = ["lag_1", "lag_2", "rolling_7_mean", "rolling_30_mean", "day_of_week", "day_of_month", "month"]

    predictions = model.predict_multiple_targets(sample_df, targets=["new_cases", "new_deaths"], feature_names=features, days_ahead=3)
    assert isinstance(predictions, dict)
    assert "new_cases" in predictions
    assert "new_deaths" in predictions
    assert predictions["new_cases"].shape[0] == 3
