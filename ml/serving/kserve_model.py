"""KServe ModelServer entrypoint — loads model from MLflow registry (e.g. @Staging)."""

from __future__ import annotations

import logging
import os

from ml.data.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)
from ml.serving.kserve_handlers import format_error, run_predict
from ml.serving.predictor import load_predictor


class ChurnKServeModel:
    """Thin wrapper so tests can import logic without installing kserve."""

    def __init__(self, name: str = "customer-churn-xgboost"):
        self.name = name
        self.predictor = None
        self.ready = False

    def load(self) -> None:
        use_local = os.environ.get("KSERVE_USE_LOCAL_JOBLIB", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if use_local:
            logger.info("KSERVE_USE_LOCAL_JOBLIB set — loading artifacts/models/churn_pipeline.joblib")
            os.environ.pop("MLFLOW_MODEL_URI", None)
        logger.info(
            "Loading model (MLFLOW_TRACKING_URI=%s, MLFLOW_MODEL_URI=%s)",
            os.environ.get("MLFLOW_TRACKING_URI", ""),
            os.environ.get("MLFLOW_MODEL_URI", ""),
        )
        try:
            self.predictor = load_predictor()
        except Exception:
            logger.exception("Model load failed")
            raise
        self.ready = True
        logger.info("Model loaded successfully")

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
            # Do not load here — ModelServer must bind HTTP before a slow MLflow download.

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
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    model_name = os.environ.get("KSERVE_MODEL_NAME", "customer-churn-xgboost")
    try:
        KServeChurnModel, ModelServer = _wrap_kserve_model()
        model = KServeChurnModel(model_name)
        ModelServer(http_port=int(os.environ.get("KSERVE_HTTP_PORT", "8080"))).start([model])
    except Exception:
        logger.exception("KServe model server failed to start")
        raise


if __name__ == "__main__":
    main()
