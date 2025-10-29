from pathlib import Path
import pickle
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TUNED = MODELS_DIR / "car_price_model_tuned.pkl"
BASE = MODELS_DIR / "car_price_model.pkl"
MODEL_PATH = TUNED if TUNED.exists() else BASE

def format_inr(amount):
    try:
        return f"₹{int(round(amount)):,}"
    except Exception:
        return str(amount)

if not MODEL_PATH.exists():
    print("No model found. Run train.py or train_tune.py first.")
    raise SystemExit(1)

print(f"Using model: {MODEL_PATH.name}")
model = pickle.load(open(MODEL_PATH,'rb'))

DATA = ROOT / "data" / "cleaned_cars.csv"
if not DATA.exists():
    print("Cleaned data not found. Run data_processing.py first.")
    raise SystemExit(1)

df = pd.read_csv(DATA)
if df.shape[0] == 0:
    print("Cleaned data is empty.")
    raise SystemExit(1)

# prepare single-row input
features = [
    "brand","model","age","km_driven","mileage","engine","max_power","owner","fuel_type","transmission","seller_type"
]
row = df.iloc[0]
X = pd.DataFrame([{f: row[f] for f in features}])

print("Input:")
print(X.to_dict(orient='records')[0])

pred = model.predict(X)[0]
print("Prediction:", format_inr(pred))
