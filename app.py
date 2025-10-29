import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

import datetime

from used_car_utils import load_pickle
import os
# Note: avoid `from datetime import datetime` to prevent name shadowing
# we will use `datetime.datetime` consistently below

ROOT = Path(__file__).resolve().parent
# prefer the better-performing model: compare tuned vs baseline on a small
# validation sample at startup (if both exist) and pick the one with lower RMSE.
TUNED_MODEL = ROOT / "models" / "car_price_model_tuned.pkl"
BASELINE_MODEL = ROOT / "models" / "car_price_model.pkl"

def _choose_best_model():
    """Return path to best model among tuned and baseline using a tiny eval set.

    If only one exists, return it. If neither exists, return tuned path (missing)
    so existing logic shows no model found.
    """
    # quick helper to evaluate a pipeline on sample data
    def _eval_rmse(pipe, df):
        try:
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
            if df is None or df.empty:
                return float('inf')
            df2 = df.copy()
            # ensure age exists
            if 'age' not in df2.columns and 'year' in df2.columns:
                df2['age'] = datetime.datetime.now().year - df2['year']
            X = df2[features].head(100)
            y = df2['selling_price'].head(100) if 'selling_price' in df2.columns else None
            preds = pipe.predict(X)
            if y is None:
                # no ground truth available; return variance of predictions as proxy
                return float(((preds - preds.mean()) ** 2).mean()) ** 0.5
            from sklearn.metrics import mean_squared_error
            return mean_squared_error(y, preds) ** 0.5
        except Exception:
            return float('inf')

    # load a small sample dataframe from cleaned or sample data
    sample_df = None
    try:
        sample_path = ROOT / 'data' / 'cleaned_cars.csv'
        if not sample_path.exists():
            sample_path = ROOT / 'data' / 'sample_cars.csv'
        if sample_path.exists():
            import pandas as _pd

            sample_df = _pd.read_csv(sample_path)
    except Exception:
        sample_df = None

    tuned_exists = TUNED_MODEL.exists()
    base_exists = BASELINE_MODEL.exists()
    if tuned_exists and base_exists:
        try:
            tuned_pipe = load_pickle(str(TUNED_MODEL))
            base_pipe = load_pickle(str(BASELINE_MODEL))
            tuned_rmse = _eval_rmse(tuned_pipe, sample_df)
            base_rmse = _eval_rmse(base_pipe, sample_df)
            # smaller is better
            return TUNED_MODEL if tuned_rmse <= base_rmse else BASELINE_MODEL
        except Exception:
            return TUNED_MODEL
    if tuned_exists:
        return TUNED_MODEL
    if base_exists:
        return BASELINE_MODEL
    return TUNED_MODEL


MODEL_PATH = _choose_best_model()
BRAND_MODELS = ROOT / "brand_models.json"


def format_inr(amount: float) -> str:
    try:
        return f"₹{int(round(amount)):,}"
    except Exception:
        return f"₹{amount}"


st.set_page_config(page_title="Used Car Price Predictor (India)", layout="centered")

st.title("Used Car Price Predictor — India")
st.write("Predict resale price in INR (₹) — enter details and submit to get an estimate.")

# Show which model file is being used
model_label = MODEL_PATH.name if MODEL_PATH.exists() else "(no model found)"
try:
    mstat = MODEL_PATH.stat()
    # use module-level datetime to avoid name conflicts
    mtime = datetime.datetime.fromtimestamp(mstat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    msize = f"{mstat.st_size:,} bytes"
    st.sidebar.info(f"Model: {model_label}\nLast updated: {mtime}\nSize: {msize}")
    # also print to stdout so server logs show which model was loaded
    try:
        print(f"Using model: {model_label} | Last updated: {mtime} | Size: {msize}")
    except Exception:
        pass
except Exception:
    st.sidebar.info(f"Model: {model_label}")

with open(BRAND_MODELS, "r", encoding="utf-8") as f:
    brand_models = json.load(f)

# Prepare session state for dynamic model options in the UI
if 'model_options' not in st.session_state:
    st.session_state['model_options'] = ["-- Select --"]
    # initialize brand/model keys so callbacks work predictably
if 'brand' not in st.session_state:
    st.session_state['brand'] = "-- Select --"
if 'model' not in st.session_state:
    st.session_state['model'] = "-- Select --"


def _update_model_options():
    """Callback to update model options when brand changes."""
    b = st.session_state.get('brand', "-- Select --")
    models = brand_models.get(b, []) if b and b != "-- Select --" else []
    opts = ["-- Select --"] + models if models else ["-- Select --"]
    st.session_state['model_options'] = opts
    # reset selected model when brand changes
    st.session_state['model'] = "-- Select --"

# Sidebar for inputs keeps main area clean
# Put brand/model selectors outside the form because callbacks inside forms
# are restricted (Streamlit only allows callbacks on the form submit button).
# Brand selectbox updates session_state['brand'] and triggers model options update
brand = st.sidebar.selectbox(
    "Brand",
    options=["-- Select --"] + sorted(list(brand_models.keys())),
    key='brand',
    on_change=_update_model_options,
)

# Model options are driven by session_state['model_options'] so they change immediately
model = st.sidebar.selectbox(
    "Model",
    options=st.session_state.get('model_options', ["-- Select --"]),
    key='model',
)

# Now the rest of the inputs live inside the form so a single submit triggers
with st.sidebar.form(key="input_form"):
    st.header("Car details")
    current_year = datetime.datetime.now().year
    year = st.number_input("Year of Manufacture", min_value=1980, max_value=current_year, value=2016)
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=40000, step=500)
    mileage = st.number_input("Mileage (kmpl)", min_value=1.0, value=18.0, step=0.1)
    engine = st.number_input("Engine (CC)", min_value=500, value=1197, step=10)
    max_power = st.number_input("Max Power (bhp)", min_value=10.0, value=80.0, step=1.0)
    owner = st.selectbox("Number of Previous Owners", options=[0, 1, 2, 3], index=1)
    fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = st.selectbox("Transmission", options=["Manual", "Automatic"])
    seller_type = st.selectbox("Seller Type", options=["Individual", "Dealer", "Trustmark Dealer"])

    submit = st.form_submit_button("Predict Price")


def validate_inputs():
    errors = []
    if brand == "-- Select --":
        errors.append("Please select a Brand.")
    if model == "-- Select --":
        errors.append("Please select a Model.")
    if year > datetime.datetime.now().year:
        errors.append("Year cannot be in the future.")
    for val, name in [(km_driven, "Kilometers Driven"), (mileage, "Mileage"), (engine, "Engine")]:
        if val <= 0:
            errors.append(f"{name} must be positive.")
    return errors


if submit:
    # If model missing, show an actionable message
    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python train.py` to train and save a model first.")
    else:
        errs = validate_inputs()
        if errs:
            for e in errs:
                st.error(e)
        else:
            # prepare single-row dataframe for prediction
            age = datetime.datetime.now().year - int(year)
            X = pd.DataFrame([
                {
                    "brand": brand,
                    "model": model,
                    "age": age,
                    "km_driven": int(km_driven),
                    "mileage": float(mileage),
                    "engine": int(engine),
                    "max_power": float(max_power),
                    "owner": int(owner),
                    "fuel_type": fuel_type,
                    "transmission": transmission,
                    "seller_type": seller_type,
                }
            ])
            try:
                model_pipe = load_pickle(str(MODEL_PATH))
                # quick sanity: if the model returns identical predictions for
                # two distinct inputs, prefer the baseline model (if present)
                try:
                    sample1 = X.copy()
                    sample2 = X.copy()
                    # perturb a numeric feature to test sensitivity
                    sample2.loc[0, 'km_driven'] = int(sample2.loc[0, 'km_driven'] * 1.1 + 100)
                    p1 = model_pipe.predict(sample1)[0]
                    p2 = model_pipe.predict(sample2)[0]
                    if np.isclose(p1, p2, rtol=1e-6, atol=1e-2):
                        # attempt to fall back to baseline model
                        baseline = ROOT / 'models' / 'car_price_model.pkl'
                        if baseline.exists() and str(baseline) != str(MODEL_PATH):
                            try:
                                alt = load_pickle(str(baseline))
                                p_alt1 = alt.predict(sample1)[0]
                                p_alt2 = alt.predict(sample2)[0]
                                if not np.isclose(p_alt1, p_alt2, rtol=1e-6, atol=1e-2):
                                    model_pipe = alt
                                else:
                                    st.warning('Model seems to produce identical outputs for different inputs — consider retraining the model with more data.')
                            except Exception:
                                st.warning('Fallback model exists but failed to load.')
                        else:
                            st.warning('Model seems to produce identical outputs for different inputs — consider retraining the model with more data.')
                except Exception:
                    # if the quick sanity check fails for any reason, continue and try prediction
                    pass
                pred = model_pipe.predict(X)[0]
                st.success(f"Predicted resale price: {format_inr(pred)}")
                st.caption("Estimate — actual market price may vary. Use this as a guide.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


st.markdown("---")
