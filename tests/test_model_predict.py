import pandas as pd
from pathlib import Path
import pickle


def test_model_predicts_numeric():
    model_path = Path(__file__).resolve().parents[1] / "models" / "car_price_model.pkl"
    assert model_path.exists(), "Model file must exist at models/car_price_model.pkl for this test"
    model = pickle.load(open(model_path, "rb"))
    age = pd.Timestamp.now().year - 2016
    X = pd.DataFrame([{
        'brand': 'Maruti',
        'model': 'Swift',
        'age': age,
        'km_driven': 45000,
        'mileage': 23.0,
        'engine': 1197,
        'max_power': 83,
        'owner': 1,
        'fuel_type': 'Petrol',
        'transmission': 'Manual',
        'seller_type': 'Individual',
    }])
    pred = model.predict(X)
    assert hasattr(pred, '__len__')
    assert float(pred[0]) > 0
