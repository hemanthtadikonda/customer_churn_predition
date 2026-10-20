from ml.data.schema import FEATURE_COLUMNS
from ml.serving.kserve_handlers import predict_v1_body, predict_v2_inputs, run_predict


def test_v1_predict_batch(trained_predictor, sample_df):
    row = sample_df.drop(columns=["customerID", "Churn"]).iloc[0].to_dict()
    body = {"instances": [row, row]}
    out = predict_v1_body(body, trained_predictor)
    assert len(out["predictions"]) == 2
    assert "churn_probability" in out["predictions"][0]


def test_v2_tensor_single_row(trained_predictor, sample_df):
    row = sample_df.drop(columns=["customerID", "Churn"]).iloc[0]
    data = [row[col] for col in FEATURE_COLUMNS]
    inputs = [
        {
            "name": "features",
            "shape": [1, len(FEATURE_COLUMNS)],
            "datatype": "FP32",
            "data": data,
        }
    ]
    out = predict_v2_inputs(inputs, trained_predictor, FEATURE_COLUMNS)
    assert out["outputs"][0]["name"] == "churn_probability"
    assert len(out["predictions"]) == 1


def test_run_predict_routes_v1(trained_predictor, sample_df):
    row = sample_df.drop(columns=["customerID", "Churn"]).iloc[0].to_dict()
    out = run_predict({"instances": [row]}, trained_predictor, FEATURE_COLUMNS)
    assert "predictions" in out
