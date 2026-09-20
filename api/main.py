from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)
from ml.config import load_config
from ml.data.schema import SchemaError
from ml.serving.predictor import ChurnPredictor, load_predictor

config = load_config()
predictor: ChurnPredictor | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global predictor
    if predictor is None:
        try:
            predictor = load_predictor()
        except FileNotFoundError:
            predictor = None
    yield


app = FastAPI(
    title=config["api"]["title"],
    version=config["project"]["version"],
    description="Score telco customers for churn risk using a frozen XGBoost pipeline.",
    lifespan=lifespan,
)


def _require_predictor() -> ChurnPredictor:
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded. Train first with: python -m ml.training.train",
        )
    return predictor


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = predictor is not None
    metadata = predictor.metadata if predictor else {}
    return HealthResponse(
        status="ok" if loaded else "model_missing",
        model_loaded=loaded,
        model_name=metadata.get("model_name"),
        algorithm=metadata.get("algorithm"),
    )


@app.get("/model")
def model_info() -> dict:
    current = _require_predictor()
    return current.metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures) -> PredictionResponse:
    current = _require_predictor()
    try:
        result = current.predict_one(payload.model_dump())
    except SchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    current = _require_predictor()
    try:
        results = current.predict_batch([row.model_dump() for row in payload.customers])
    except SchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BatchPredictionResponse(predictions=[PredictionResponse(**row) for row in results])
