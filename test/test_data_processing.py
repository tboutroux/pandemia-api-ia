import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pandas as pd
import pytest
from app.ai.data_processing import create_features

@pytest.fixture
def sample_df():
    date_range = pd.date_range(start="2020-01-01", periods=40, freq="D")
    data = {
        "new_cases": range(40),
        "new_deaths": range(0, 80, 2),
        "new_recovered": range(0, 120, 3),
        "population": [1000000] * 40
    }
    df = pd.DataFrame(data, index=date_range)
    return df

def test_create_features_all_enabled(sample_df):
    df_features = create_features(sample_df.copy(), target="new_cases")
    assert "lag_1" in df_features.columns
    assert "rolling_7_mean" in df_features.columns
    assert "day_of_week" in df_features.columns
    assert "cases_per_100k" in df_features.columns
    assert not df_features.isnull().any().any()

def test_create_features_disable_all(sample_df):
    df_features = create_features(sample_df.copy(), target="new_cases", use_lags=False, use_rolling=False, use_calendar=False)
    assert "lag_1" not in df_features.columns
    assert "rolling_7_mean" not in df_features.columns
    assert "day_of_week" not in df_features.columns

def test_create_features_partial(sample_df):
    df_features = create_features(sample_df.copy(), target="new_cases", use_lags=True, use_rolling=False, use_calendar=False)
    assert "lag_1" in df_features.columns
    assert "rolling_7_mean" not in df_features.columns
    assert "day_of_week" not in df_features.columns
