"""
Hyperparameter tuning script for RandomForest and XGBoost regressors.

Saves:
- models/car_price_model_tuned.pkl (best pipeline)
- models/tuning_results.csv (grid/search results)

This is intentionally conservative (small n_iter) so it runs quickly on sample data.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    has_xgb = True
except Exception:
    has_xgb = False

from used_car_utils import save_pickle

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cars.csv"
SAMPLE_PATH = ROOT / "data" / "sample_cars.csv"
CLEANED_PATH = ROOT / "data" / "cleaned_cars.csv"
MODELS_DIR = ROOT / "models"


def load_data():
    if CLEANED_PATH.exists():
        df = pd.read_csv(CLEANED_PATH)
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(SAMPLE_PATH)
    return df


def preprocess_df(df):
    # small inline cleaning to match train.py expectations
    df = df.copy()
    # ensure numeric coercion similar to data_processing
    for c in ["year", "km_driven", "mileage", "engine", "max_power", "owner", "selling_price"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(r"[^0-9\.]+", "", regex=True)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "year" in df.columns:
        df["age"] = pd.Timestamp.now().year - df["year"]
    df = df.dropna()
    return df


def build_pipeline(categorical_features, numeric_features):
    numeric_transformer = StandardScaler()
    # handle sklearn API differences for sparse output parameter
    try:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def main():
    df = load_data()
    df = preprocess_df(df)
    features = [
        "brand",
        "model",
        "age",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "owner",
        "fuel_type",
        "transmission",
        "seller_type",
    ]
    for f in features:
        if f not in df.columns:
            raise ValueError(f"Expected feature column '{f}' in data but not found.")

    X = df[features]
    y = df["selling_price"]

    numeric_features = ["age", "km_driven", "mileage", "engine", "max_power", "owner"]
    categorical_features = [c for c in features if c not in numeric_features]

    preprocessor = build_pipeline(categorical_features, numeric_features)

    # Candidate estimators and parameter distributions
    rf = RandomForestRegressor(random_state=42)
    rf_param_dist = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
    }

    candidates = [("RandomForest", rf, rf_param_dist)]
    if has_xgb:
        xgb = XGBRegressor(random_state=42, verbosity=0, objective="reg:squarederror")
        xgb_param_dist = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
        }
        candidates.append(("XGBoost", xgb, xgb_param_dist))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tuning_results = []
    best_overall = None
    best_rmse = float("inf")

    for name, estimator, param_dist in candidates:
        print(f"Tuning {name}...")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        # small budget to keep runtime low
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=6,
            cv=3,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=1,
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)

        best = search.best_estimator_
        preds = best.predict(X_test)
        # compute RMSE compatibly across sklearn versions
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        print(f"Best {name}: RMSE={rmse:.2f}, R2={r2:.3f}")

        tuning_results.append({"model": name, "rmse": float(rmse), "r2": float(r2), "best_params": str(search.best_params_)})

        # persist best per-candidate
        candidate_path = MODELS_DIR / f"car_price_model_{name.lower()}.pkl"
        save_pickle(best, str(candidate_path))

        if rmse < best_rmse:
            best_rmse = rmse
            best_overall = (name, best)

    # save tuning results
    results_df = pd.DataFrame(tuning_results)
    results_csv = MODELS_DIR / "tuning_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Saved tuning results to {results_csv}")

    if best_overall:
        name, model_pipe = best_overall
        out_path = MODELS_DIR / "car_price_model_tuned.pkl"
        save_pickle(model_pipe, str(out_path))
        print(f"Saved best tuned model ({name}) to {out_path}")


if __name__ == "__main__":
    main()
