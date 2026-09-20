from ml.serving.predictor import risk_band


def test_risk_bands():
    assert risk_band(0.12) == "low"
    assert risk_band(0.30) == "medium"
    assert risk_band(0.59) == "medium"
    assert risk_band(0.60) == "high"


def test_predictor_scores_a_customer(trained_predictor, sample_df):
    payload = sample_df.drop(columns=["customerID", "Churn"]).iloc[0].to_dict()
    result = trained_predictor.predict_one(payload)
    assert set(result) >= {
        "churn_probability",
        "churn_prediction",
        "risk_band",
        "threshold",
        "algorithm",
    }
    assert 0.0 <= result["churn_probability"] <= 1.0
