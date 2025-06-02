import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pytest
import pandas as pd
from unittest.mock import MagicMock
from sqlalchemy.engine.base import Engine
from app.ai.database import create_db_engine, load_data

def test_create_db_engine_url():
    engine = create_db_engine("user", "pass", "localhost", "mydb")
    assert isinstance(engine, Engine)
    assert "mysql+pymysql" in str(engine.url)

def test_load_data_valid_targets(monkeypatch):
    mock_engine = MagicMock()

    def mock_read_sql(query, engine, params):
        return pd.DataFrame({
            'date': pd.date_range("2023-01-01", periods=3),
            'population': [1000, 1000, 1000],
            'new_cases': [10, 20, 30],
            'new_deaths': [1, 2, 3],
            'new_recovered': [5, 10, 15],
        })

    monkeypatch.setattr(pd, "read_sql", mock_read_sql)
    df = load_data(mock_engine, "France")
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)

def test_load_data_custom_targets(monkeypatch):
    mock_engine = MagicMock()

    def mock_read_sql(query, engine, params):
        return pd.DataFrame({
            'date': pd.date_range("2023-01-01", periods=2),
            'population': [1000, 1000],
            'new_cases': [10, 20],
        })

    monkeypatch.setattr(pd, "read_sql", mock_read_sql)
    df = load_data(mock_engine, "Germany", targets=["new_cases"])
    assert "new_cases" in df.columns
    assert "new_deaths" not in df.columns

def test_load_data_invalid_target():
    with pytest.raises(ValueError, match="Target 'invalid' invalide"):
        load_data(MagicMock(), "Spain", targets=["invalid"])

def test_load_data_empty_df(monkeypatch):
    mock_engine = MagicMock()

    def mock_read_sql(query, engine, params):
        return pd.DataFrame()

    monkeypatch.setattr(pd, "read_sql", mock_read_sql)

    with pytest.raises(ValueError, match="Aucune donnée trouvée pour"):
        load_data(mock_engine, "Italy")

def test_load_data_sql_exception(monkeypatch, caplog):
    mock_engine = MagicMock()

    def failing_read_sql(query, engine, params):
        raise Exception("SQL error")

    monkeypatch.setattr(pd, "read_sql", failing_read_sql)

    with pytest.raises(Exception, match="SQL error"):
        load_data(mock_engine, "Brazil")

    assert "Erreur lors du chargement des données" in caplog.text
