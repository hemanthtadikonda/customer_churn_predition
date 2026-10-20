"""KServe request/response helpers (no kserve dependency — safe for unit tests)."""

from __future__ import annotations

from typing import Any

from ml.data.schema import SchemaError
from ml.serving.predictor import ChurnPredictor


def predict_v1_body(body: dict[str, Any], predictor: ChurnPredictor) -> dict[str, Any]:
    """Cloud Events / KServe V1 JSON: ``{"instances": [{...}, ...]}``."""
    instances = body.get("instances")
    if instances is None:
        raise ValueError('Request body must include an "instances" array.')
    if not isinstance(instances, list):
        raise ValueError('"instances" must be a list of customer feature objects.')
    if not instances:
        raise ValueError('"instances" must not be empty.')
    predictions = predictor.predict_batch(instances)
    return {"predictions": predictions}


def _tensor_row_to_dict(row: list[Any], names: list[str]) -> dict[str, Any]:
    if len(row) != len(names):
        raise ValueError(f"Expected {len(names)} feature values, got {len(row)}.")
    return dict(zip(names, row))


def predict_v2_inputs(
    inputs: list[dict[str, Any]],
    predictor: ChurnPredictor,
    feature_names: list[str],
) -> dict[str, Any]:
    """KServe V2: tabular features as one tensor input named ``features``."""
    if not inputs:
        raise ValueError("V2 request must include at least one input.")
    tensor_input = next((item for item in inputs if item.get("name") == "features"), inputs[0])
    data = tensor_input.get("data")
    if data is None:
        raise ValueError('Input "features" is missing "data".')
    shape = tensor_input.get("shape")
    datatype = str(tensor_input.get("datatype", "FP32")).upper()

    rows: list[dict[str, Any]]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        rows = data
    elif isinstance(data, list):
        if shape and len(shape) == 2:
            n_cols = int(shape[1])
            if n_cols != len(feature_names):
                raise ValueError(
                    f"Tensor width {n_cols} does not match {len(feature_names)} feature columns."
                )
            rows = [_tensor_row_to_dict(list(row), feature_names) for row in data]
        else:
            if len(data) != len(feature_names):
                raise ValueError(
                    f"Expected {len(feature_names)} scalar values for one row, got {len(data)}."
                )
            rows = [_tensor_row_to_dict(data, feature_names)]
    else:
        raise ValueError("Unsupported V2 input data format.")

    if datatype not in {"FP32", "FP64", "BYTES", "BOOL"}:
        raise ValueError(f"Unsupported datatype: {datatype}")

    predictions = predictor.predict_batch(rows)
    probabilities = [float(row["churn_probability"]) for row in predictions]
    labels = [int(row["churn_prediction"]) for row in predictions]
    return {
        "model_name": predictor.metadata.get("model_name", "churn"),
        "outputs": [
            {
                "name": "churn_probability",
                "shape": [len(probabilities), 1],
                "datatype": "FP32",
                "data": probabilities,
            },
            {
                "name": "churn_prediction",
                "shape": [len(labels), 1],
                "datatype": "INT32",
                "data": labels,
            },
        ],
        "predictions": predictions,
    }


def run_predict(
    payload: dict[str, Any],
    predictor: ChurnPredictor,
    feature_names: list[str],
) -> dict[str, Any]:
    """Route V1 (instances) vs V2 (inputs) payloads."""
    if "instances" in payload:
        return predict_v1_body(payload, predictor)
    if "inputs" in payload:
        return predict_v2_inputs(payload.get("inputs") or [], predictor, feature_names)
    raise ValueError('Request must include either "instances" (V1) or "inputs" (V2).')


def format_error(exc: BaseException) -> tuple[int, str]:
    if isinstance(exc, SchemaError):
        return 422, str(exc)
    if isinstance(exc, ValueError):
        return 400, str(exc)
    return 500, str(exc)
