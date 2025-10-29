from pathlib import Path
import sys
import pandas as pd

# ensure project root is on sys.path so imports work when running this script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from used_car_utils import load_pickle
TUNED = ROOT / 'models' / 'car_price_model_tuned.pkl'
BASE = ROOT / 'models' / 'car_price_model.pkl'
MODEL_PATH = TUNED if TUNED.exists() else BASE
print('Using model path:', MODEL_PATH)

model_pipe = load_pickle(str(MODEL_PATH))
model = model_pipe.named_steps.get('model', None)
print('Loaded model class:', type(model))
if hasattr(model, 'feature_importances_'):
    print('feature_importances_ (first 10):', getattr(model, 'feature_importances_')[:10])
if hasattr(model, 'coef_'):
    print('coef_ (first 10):', getattr(model, 'coef_')[:10])
if hasattr(model, 'intercept_'):
    print('intercept_:', getattr(model, 'intercept_'))

rows = [
    {"brand":"Toyota","model":"Corolla","age":9,"km_driven":37000,"mileage":18.0,"engine":1197,"max_power":80.0,"owner":1,"fuel_type":"Petrol","transmission":"Automatic","seller_type":"Individual"},
    {"brand":"Mahindra","model":"Scorpio","age":8,"km_driven":40000,"mileage":12.0,"engine":2498,"max_power":118.0,"owner":1,"fuel_type":"Diesel","transmission":"Manual","seller_type":"Individual"},
]
X = pd.DataFrame(rows)
print('Input:')
print(X)

preds = model_pipe.predict(X)
print('Predictions:')
print(preds)
# Emulate app fallback behavior
try:
    s1 = X.copy()
    s2 = X.copy()
    s2.loc[0, 'km_driven'] = int(s2.loc[0, 'km_driven'] * 1.1 + 100)
    p1 = model_pipe.predict(s1)[0]
    p2 = model_pipe.predict(s2)[0]
    print('Sanity check p1, p2:', p1, p2)
    if abs(p1 - p2) < 1e-2:
        baseline = ROOT / 'models' / 'car_price_model.pkl'
        if baseline.exists() and str(baseline) != str(MODEL_PATH):
            alt = load_pickle(str(baseline))
            p_alt1 = alt.predict(s1)[0]
            p_alt2 = alt.predict(s2)[0]
            print('Baseline predictions p_alt1, p_alt2:', p_alt1, p_alt2)
        else:
            print('No baseline model available or same as tuned.')
except Exception as e:
    print('Sanity check failed:', e)
