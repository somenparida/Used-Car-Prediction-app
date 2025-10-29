"""
Train a used-car price model. Uses `data/sample_cars.csv` by default.

Saves model and preprocessing pipeline to `models/car_price_model.pkl` and `models/preprocessor.pkl`.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import datetime
import zoneinfo

from sklearn.model_selection import train_test_split
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

CURRENT_YEAR = pd.Timestamp.now().year

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cars.csv"
SAMPLE_PATH = ROOT / "data" / "sample_cars.csv"
CLEANED_PATH = ROOT / "data" / "cleaned_cars.csv"
MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.csv"

def load_data() -> pd.DataFrame:
    # prefer cleaned file if available
    if CLEANED_PATH.exists():
        print(f"Loading cleaned dataset from {CLEANED_PATH}")
        df = pd.read_csv(CLEANED_PATH)
    elif DATA_PATH.exists():
        print(f"Loading dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("No dataset at data/cars.csv — using sample dataset.")
        df = pd.read_csv(SAMPLE_PATH)
    return df

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    # basic cleaning and feature engineering
    df = df.copy()
    # Ensure correct dtypes
    numeric_cols = ["year", "km_driven", "mileage", "engine", "max_power", "owner"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # compute age
    if "year" in df.columns:
        df["age"] = CURRENT_YEAR - df["year"]

    # drop rows with missing target
    df = df.dropna(subset=["selling_price"]) if "selling_price" in df.columns else df

    # drop rows with too many NAs
    df = df.dropna()
    return df

def build_pipeline(categorical_features, numeric_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor

def train():
    df = load_data()
    df = preprocess_df(df)

    # feature columns
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

    # Models to evaluate
    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    if has_xgb:
        candidates["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = None
    best_rmse = float("inf")
    best_name = None
    metrics = []

    for name, model in candidates.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        # older sklearn versions may not support `squared` kwarg; compute RMSE directly
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        print(f"{name} RMSE: {rmse:.2f}, R2: {r2:.3f}")
        metrics.append({"model": name, "rmse": float(rmse), "r2": float(r2)})
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline
            best_name = name

    print(f"Best model: {best_name} with RMSE={best_rmse:.2f}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "car_price_model.pkl"
    save_pickle(best_model, str(model_path))
    # persist metrics
    try:
        # use timezone-aware UTC timestamps to avoid deprecation warnings
        utc_now = datetime.datetime.now(tz=zoneinfo.ZoneInfo('UTC'))
        meta = {
            "timestamp": utc_now.isoformat(),
            "model": best_name,
            "rmse": float(best_rmse),
            "r2": float(r2),
            "rows": int(len(df)),
        }
        metrics_df = pd.DataFrame(metrics)
        metrics_df["run_time_utc"] = utc_now.isoformat()
        # if metrics csv exists, append; otherwise create
        if METRICS_PATH.exists():
            metrics_df.to_csv(METRICS_PATH, mode="a", header=False, index=False)
        else:
            metrics_df.to_csv(METRICS_PATH, index=False)
        print(f"Saved metrics to {METRICS_PATH}")
    except Exception as e:
        print(f"Failed to save metrics: {e}")
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    train()
