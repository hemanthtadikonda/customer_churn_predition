from pathlib import Path

import pytest
from xgboost import XGBClassifier

from ml.data.generate import generate_churn_dataset
from ml.data.ingest import split_features_target, train_val_test_split
from ml.features.engineer import build_feature_pipeline
from ml.serving.predictor import ChurnPredictor
from sklearn.pipeline import Pipeline


@pytest.fixture
def sample_df():
    return generate_churn_dataset(n_samples=400, random_state=7)


@pytest.fixture
def trained_predictor(sample_df):
    features, target, _ids = split_features_target(sample_df)
    x_train, _x_val, _x_test, y_train, _y_val, _y_test = train_val_test_split(
        features,
        target,
        test_size=0.2,
        val_size=0.2,
        random_state=7,
    )
    pipeline = Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            (
                "model",
                XGBClassifier(
                    n_estimators=25,
                    max_depth=3,
                    learning_rate=0.2,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=1,
                    random_state=7,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    return ChurnPredictor(
        pipeline=pipeline,
        metadata={"model_name": "xgboost_churn_test", "algorithm": "XGBoost"},
        threshold=0.5,
    )


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
