from ml.data.ingest import split_features_target
from ml.data.schema import DERIVED_NUMERIC_FEATURES
from ml.features.engineer import ChurnFeatureEngineer, build_feature_pipeline


def test_feature_engineer_adds_derived_columns(sample_df):
    features, _target, _ids = split_features_target(sample_df)
    transformed = ChurnFeatureEngineer().fit_transform(features)
    for column in DERIVED_NUMERIC_FEATURES:
        assert column in transformed.columns
    assert transformed["services_count"].between(0, 6).all()
    assert transformed["TotalCharges"].dtype.kind in "fc"
    assert transformed.isna().sum().sum() == 0


def test_feature_pipeline_is_stable_for_serving(sample_df):
    features, _target, _ids = split_features_target(sample_df)
    pipeline = build_feature_pipeline()
    train_matrix = pipeline.fit_transform(features.iloc[:250])
    serve_matrix = pipeline.transform(features.iloc[250:260])
    assert train_matrix.shape[1] == serve_matrix.shape[1]
    assert serve_matrix.shape[0] == 10
