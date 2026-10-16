import os

import pytest

from ml.config import load_config
from ml.mlflow_utils import (
    registry_settings,
    resolve_serving_model_uri,
    resolve_tracking_uri,
    tracking_server_required_for_registry,
)


def test_resolve_serving_model_uri_prefers_env(monkeypatch):
    config = load_config()
    monkeypatch.setenv("MLFLOW_MODEL_URI", "models:/customer-churn-xgboost@Production")
    assert resolve_serving_model_uri(config) == "models:/customer-churn-xgboost@Production"


def test_resolve_serving_model_uri_falls_back_to_config(monkeypatch):
    config = load_config()
    monkeypatch.delenv("MLFLOW_MODEL_URI", raising=False)
    assert resolve_serving_model_uri(config) == config["mlflow"].get("model_uri") or None


def test_tracking_server_required_for_registry():
    assert tracking_server_required_for_registry("http://127.0.0.1:5000")
    assert not tracking_server_required_for_registry("file:///tmp/mlruns")


def test_registry_settings_env_override_disable(monkeypatch):
    config = load_config()
    monkeypatch.setenv("MLFLOW_REGISTER_MODEL", "false")
    settings = registry_settings(config)
    assert settings["enabled"] is False


def test_resolve_tracking_uri_env(monkeypatch):
    config = load_config()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    assert resolve_tracking_uri(config) == "http://127.0.0.1:5000"
