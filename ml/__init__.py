"""Customer Churn Prediction — ML package.

Layout maps to a typical industry handoff:
- ml.data: collect / generate / validate a training dataset
- ml.features: feature engineering that ships with the model
- ml.training: XGBoost training, evaluation, and artifact writes
- ml.serving: load the frozen pipeline and score customers
"""

from ml.config import PROJECT_ROOT, load_config

__all__ = ["PROJECT_ROOT", "load_config"]
