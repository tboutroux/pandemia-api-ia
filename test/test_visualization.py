import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import os
import pytest
from app.ai.visualization import (
    plot_predictions,
    plot_combined_predictions,
    plot_mortality_rate,
    plot_recovery_rate,
    save_metrics,
    plot_residuals,
    visualize_all_results,
)

@pytest.fixture
def sample_prediction_data():
    dates = pd.date_range("2023-01-01", periods=10)
    df = pd.DataFrame({
        "new_cases": range(10),
        "new_deaths": range(0, 20, 2),
        "new_recovered": range(0, 30, 3)
    }, index=dates)

    preds = {
        "new_cases": pd.DataFrame({"predicted_new_cases": range(10, 15)}, index=pd.date_range("2023-01-11", periods=5)),
        "new_deaths": pd.DataFrame({"predicted_new_deaths": range(5, 10)}, index=pd.date_range("2023-01-11", periods=5)),
        "new_recovered": pd.DataFrame({"predicted_new_recovered": range(3, 8)}, index=pd.date_range("2023-01-11", periods=5))
    }

    return df, preds

def test_plot_predictions_creates_file(tmp_path, sample_prediction_data):
    df, preds = sample_prediction_data
    plot_predictions(df, preds["new_cases"], "new_cases", "France", output_dir=tmp_path)
    assert any("France_new_cases_predictions.png" in f.name for f in tmp_path.iterdir())

def test_visualize_all_results(sample_prediction_data, tmp_path):
    df, preds = sample_prediction_data
    visualize_all_results(df, preds, "France", output_dir=tmp_path)
    expected_files = [
        "France_new_cases_predictions.png",
        "France_new_deaths_predictions.png",
        "France_new_recovered_predictions.png",
        "France_combined_predictions.png",
        "France_mortality_rate.png",
        "France_mortality_stats.txt",
        "France_recovery_rate.png",
        "France_recovery_stats.txt"
    ]
    found_files = [f.name for f in tmp_path.iterdir()]
    for fname in expected_files:
        assert fname in found_files
