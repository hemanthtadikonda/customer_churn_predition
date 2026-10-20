"""KServe ModelServer entrypoint — loads model from MLflow registry (e.g. @Staging)."""

from __future__ import annotations

import os

from ml.data.schema import FEATURE_COLUMNS
from ml.serving.kserve_handlers import format_error, run_predict
from ml.serving.predictor import load_predictor


class ChurnKServeModel:
    """Thin wrapper so tests can import logic without installing kserve."""

    def __init__(self, name: str = "customer-churn-xgboost"):
        self.name = name
        self.predictor = None
        self.ready = False

    def load(self) -> None:
        self.predictor = load_predictor()
        self.ready = True

    def predict(self, payload: dict, headers: dict | None = None) -> dict:
        if not self.ready or self.predictor is None:
            raise RuntimeError("Model is not loaded.")
        return run_predict(payload, self.predictor, FEATURE_COLUMNS)


def _wrap_kserve_model():
    from kserve import Model, ModelServer
    from kserve.errors import InvalidInput

    class _KServeChurnModel(Model):
        def __init__(self, name: str):
            super().__init__(name)
            self._inner = ChurnKServeModel(name)
            self.load()

        def load(self) -> bool:
            self._inner.load()
            self.ready = True
            return self.ready

        def predict(self, payload: dict, headers: dict | None = None) -> dict:
            try:
                return self._inner.predict(payload, headers)
            except Exception as exc:
                status, message = format_error(exc)
                if status < 500:
                    raise InvalidInput(message) from exc
                raise

    return _KServeChurnModel, ModelServer


def main() -> None:
    model_name = os.environ.get("KSERVE_MODEL_NAME", "customer-churn-xgboost")
    KServeChurnModel, ModelServer = _wrap_kserve_model()
    model = KServeChurnModel(model_name)
    ModelServer(http_port=int(os.environ.get("KSERVE_HTTP_PORT", "8080"))).start([model])


if __name__ == "__main__":
    main()
