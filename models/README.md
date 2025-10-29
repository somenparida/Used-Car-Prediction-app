Models and metrics
==================

This folder stores trained model artifacts and evaluation logs.

- `car_price_model.pkl`: Pickled scikit-learn Pipeline containing the preprocessor and the selected regression model. Used by `app.py` for prediction.
- `metrics.csv`: Evaluation rows written by `train.py` when training finishes. Structure (one row per candidate model per run):

  - model: model name (e.g., LinearRegression, RandomForest, XGBoost)
  - rmse: root mean squared error on the test split
  - r2: R² score on the test split
  - run_time_utc: ISO timestamp when the run was logged

Example:

model,rmse,r2,run_time_utc
LinearRegression,48346.73,-0.908,2025-10-29T12:34:56.789012
RandomForest,199821.63,-31.595,2025-10-29T12:34:56.789012

Use this file to compare models and track experiment history.
