import pytest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
from fastapi import HTTPException, Query
from app.routers.prediction import predict, router

# Données de test
@pytest.fixture
def mock_df():
    return pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=100),
        'new_cases': np.random.randint(0, 1000, size=100),
        'new_deaths': np.random.randint(0, 100, size=100),
        'new_recovered': np.random.randint(0, 500, size=100),
        'population': [1000000] * 100
    })

@pytest.fixture
def mock_df_features():
    df = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=100),
        'new_cases': np.random.randint(0, 1000, size=100),
        'lag_1_new_cases': np.random.randint(0, 1000, size=100),
        'rolling_7_new_cases': np.random.randint(0, 1000, size=100),
        'day_of_week': np.random.randint(0, 7, size=100),
        'day_of_month': np.random.randint(1, 31, size=100),
        'month': np.random.randint(1, 13, size=100),
        'cases_per_100k': np.random.random(size=100) * 100,
    })
    return df

@pytest.fixture
def mock_predictions():
    return pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=7),
        'predicted_new_cases': np.random.randint(0, 1000, size=7)
    })

@pytest.fixture
def mock_predictions_all():
    # DataFrame avec toutes les colonnes de prédiction attendues
    return pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=7),
        'predicted_new_cases': np.random.randint(0, 1000, size=7),
        'predicted_new_deaths': np.random.randint(0, 100, size=7),
        'predicted_new_recovered': np.random.randint(0, 500, size=7)
    })

@pytest.fixture
def mock_credentials():
    return MagicMock()

@pytest.mark.parametrize("country,days", [("France", 7), ("Italy", 10)])
@patch('app.routers.prediction.create_db_engine')
@patch('app.routers.prediction.load_data')
@patch('app.routers.prediction.create_features')
@patch('app.routers.prediction.PandemicModel')
@patch('app.routers.prediction.visualize_all_results')
@patch('app.routers.prediction.os.path.exists')
@patch('app.routers.prediction.pd.DataFrame.to_csv')
def test_predict_basic(mock_to_csv, mock_exists, mock_visualize, mock_model_class, 
                      mock_create_features, mock_load_data, mock_create_db_engine, 
                      country, days, mock_df, mock_df_features, mock_predictions_all, mock_credentials):
    mock_exists.return_value = True
    mock_create_db_engine.return_value = "mock_engine"
    mock_load_data.return_value = mock_df
    mock_create_features.return_value = mock_df_features

    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    # Toujours retourner un tuple pour train_model
    mock_model.train_model.return_value = ("mock_model", {"r2": 0.8, "mse": 10})
    # Retourner toutes les colonnes de prédiction attendues
    mock_model.predict_future.return_value = mock_predictions_all
    mock_to_csv.return_value = None

    with patch('app.routers.prediction.Query', autospec=True) as mock_query:
        mock_query_instance = MagicMock()
        mock_query_instance.split.return_value = ["new_cases", "new_deaths", "new_recovered"]
        mock_query.return_value = mock_query_instance

        result = predict(country=country, days=days, targets="new_cases,new_deaths,new_recovered", credentials=mock_credentials)

    assert result["country"] == country
    assert result["days_ahead"] == days
    assert "predictions" in result
    assert "metrics" in result

@patch('app.routers.prediction.create_db_engine')
@patch('app.routers.prediction.load_data')
@patch('app.routers.prediction.create_features')
@patch('app.routers.prediction.PandemicModel')
@patch('app.routers.prediction.visualize_all_results')
@patch('app.routers.prediction.os.path.exists')
@patch('app.routers.prediction.pd.DataFrame.to_csv')
def test_predict_with_custom_targets(mock_to_csv, mock_exists, mock_visualize, mock_model_class, 
                                   mock_create_features, mock_load_data, mock_create_db_engine, 
                                   mock_df, mock_df_features, mock_predictions_all, mock_credentials):
    mock_exists.return_value = True
    mock_create_db_engine.return_value = "mock_engine"
    mock_load_data.return_value = mock_df
    mock_create_features.return_value = mock_df_features
    mock_to_csv.return_value = None

    # Retourner un tuple pour chaque appel à train_model
    mock_model = mock_model_class.return_value
    mock_model.train_model.return_value = ("mock_model", {"r2": 0.8, "mse": 10})

    # Retourner un DataFrame avec toutes les colonnes pour chaque target
    def predict_future_side_effect(df, target, **kwargs):
        return mock_predictions_all[["date", f"predicted_{target}"]].copy()
    mock_model.predict_future.side_effect = predict_future_side_effect

    with patch('app.routers.prediction.Query', autospec=True) as mock_query:
        mock_query_instance = MagicMock()
        mock_query_instance.split.return_value = ["new_cases", "new_deaths"]
        mock_query.return_value = mock_query_instance

        result = predict(country="Italy", days=10, targets="new_cases,new_deaths", credentials=mock_credentials)

    assert result["country"] == "Italy"
    assert result["days_ahead"] == 10

    mock_load_data.assert_called_once()
    call_args = mock_load_data.call_args[0]
    assert call_args[0] == "mock_engine"
    assert call_args[1] == "Italy"

    call_kwargs = mock_load_data.call_args[1]
    assert "targets" in call_kwargs
    assert call_kwargs["targets"] == ["new_cases", "new_deaths"]

@patch('app.routers.prediction.create_db_engine')
@patch('app.routers.prediction.load_data')
@patch('app.routers.prediction.create_features')
@patch('app.routers.prediction.PandemicModel')
@patch('app.routers.prediction.visualize_all_results')
@patch('app.routers.prediction.os.path.exists')
@patch('app.routers.prediction.pd.DataFrame.to_csv')
def test_predict_no_train(mock_to_csv, mock_exists, mock_visualize, mock_model_class, 
                        mock_create_features, mock_load_data, mock_create_db_engine, 
                        mock_df, mock_df_features, mock_predictions_all, mock_credentials):
    mock_exists.return_value = True
    mock_create_db_engine.return_value = "mock_engine"
    mock_load_data.return_value = mock_df
    mock_create_features.return_value = mock_df_features
    mock_to_csv.return_value = None

    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    mock_model.load_model.return_value = "existing_model"
    mock_model.predict_future.return_value = mock_predictions_all

    with patch('app.routers.prediction.Query', autospec=True) as mock_query:
        mock_query_instance = MagicMock()
        mock_query_instance.split.return_value = ["new_cases", "new_deaths", "new_recovered"]
        mock_query.return_value = mock_query_instance

        result = predict(country="Germany", days=5, no_train=True, targets="new_cases,new_deaths,new_recovered", credentials=mock_credentials)

    assert result["country"] == "Germany"
    assert result["days_ahead"] == 5

    mock_model.load_model.assert_called()

@patch('app.routers.prediction.create_db_engine')
@patch('app.routers.prediction.load_data')
@patch('app.routers.prediction.create_features')
@patch('app.routers.prediction.PandemicModel')
@patch('app.routers.prediction.visualize_all_results')
@patch('app.routers.prediction.os.path.exists')
@patch('app.routers.prediction.pd.DataFrame.to_csv')
def test_predict_with_tuning(mock_to_csv, mock_exists, mock_visualize, mock_model_class, 
                           mock_create_features, mock_load_data, mock_create_db_engine, 
                           mock_df, mock_df_features, mock_predictions_all, mock_credentials):
    mock_exists.return_value = True
    mock_create_db_engine.return_value = "mock_engine"
    mock_load_data.return_value = mock_df
    mock_create_features.return_value = mock_df_features
    mock_to_csv.return_value = None

    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    mock_model.train_model.return_value = ("mock_model", {"r2": 0.9, "mse": 5})
    mock_model.predict_future.return_value = mock_predictions_all

    with patch('app.routers.prediction.Query', autospec=True) as mock_query:
        mock_query_instance = MagicMock()
        mock_query_instance.split.return_value = ["new_cases", "new_deaths", "new_recovered"]
        mock_query.return_value = mock_query_instance

        result = predict(country="Spain", days=7, tune=True, targets="new_cases,new_deaths,new_recovered", credentials=mock_credentials)

    assert result["country"] == "Spain"
    mock_model.train_model.assert_called()
    assert mock_model.train_model.call_args[1]['tune_hyperparams'] == True

@patch('app.routers.prediction.create_db_engine')
@patch('app.routers.prediction.load_data')
@patch('app.routers.prediction.create_features')
@patch('app.routers.prediction.PandemicModel')
@patch('app.routers.prediction.visualize_all_results')
@patch('app.routers.prediction.os.path.exists')
@patch('app.routers.prediction.pd.DataFrame.to_csv')
def test_predict_no_train_model_not_found(mock_to_csv, mock_exists, mock_visualize, 
                                        mock_model_class, mock_create_features, 
                                        mock_load_data, mock_create_db_engine, 
                                        mock_df, mock_df_features, mock_predictions_all, mock_credentials):
    mock_exists.return_value = True
    mock_create_db_engine.return_value = "mock_engine"
    mock_load_data.return_value = mock_df
    mock_create_features.return_value = mock_df_features
    mock_to_csv.return_value = None

    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    mock_model.load_model.return_value = None
    mock_model.train_model.return_value = ("mock_model", {"r2": 0.8, "mse": 10})
    mock_model.predict_future.return_value = mock_predictions_all

    with patch('app.routers.prediction.Query', autospec=True) as mock_query:
        mock_query_instance = MagicMock()
        mock_query_instance.split.return_value = ["new_cases", "new_deaths", "new_recovered"]
        mock_query.return_value = mock_query_instance

        result = predict(country="Germany", days=5, no_train=True, targets="new_cases,new_deaths,new_recovered", credentials=mock_credentials)

    assert result["country"] == "Germany"
    mock_model.load_model.assert_called()
    mock_model.train_model.assert_called()
