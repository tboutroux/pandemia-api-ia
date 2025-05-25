from fastapi.testclient import TestClient
from app.main import app
from app.config.settings import API_KEY, API_KEY_NAME

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_with_valid_api_key():
    headers = {'Authorization': f'Bearer {API_KEY}'}
    response = client.get("/connect/test", headers=headers)
    # On attend un 200 si la clé est bonne, ou une autre erreur métier (pas 403)
    assert response.status_code != 403

def test_predict_with_invalid_api_key():
    headers = {API_KEY_NAME: "wrong_key"}
    response = client.get("/connect/test", headers=headers)
    assert response.status_code == 403