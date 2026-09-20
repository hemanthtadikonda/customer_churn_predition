import json

from fastapi.testclient import TestClient

from api.main import app
import api.main as api_main
from ml.data.ingest import split_features_target
from ml.data.schema import FEATURE_COLUMNS


def test_health_reports_missing_model_without_artifact(monkeypatch):
    api_main.predictor = None

    def _missing():
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(api_main, "load_predictor", _missing)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_returns_risk_band(trained_predictor, sample_df):
    api_main.predictor = trained_predictor
    features, _target, _ids = split_features_target(sample_df)
    payload = json.loads(features.iloc[0][FEATURE_COLUMNS].to_json())
    client = TestClient(app)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["algorithm"] == "XGBoost"


def test_predict_rejects_invalid_contract(trained_predictor, sample_df):
    api_main.predictor = trained_predictor
    features, _target, _ids = split_features_target(sample_df)
    payload = json.loads(features.iloc[0][FEATURE_COLUMNS].to_json())
    payload["Contract"] = "Weekly"
    client = TestClient(app)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
