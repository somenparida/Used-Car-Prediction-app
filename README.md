# Used Car Price Prediction (India)

This project is a full-stack data science app for predicting used car resale prices in INR (₹). It includes a training pipeline and a Streamlit-based UI.

What's included
- `data/sample_cars.csv` - small sample dataset for quick testing
- `train.py` - training pipeline (uses sample data by default; you can provide a real dataset)
- `app.py` - Streamlit web app to make predictions
- `models/` - model artifacts will be saved here after training
- `requirements.txt` - Python dependencies

Model artifacts and metrics
--------------------------

After training, the pipeline and metadata are saved under `models/`:

- `models/car_price_model.pkl` — the serialized scikit-learn Pipeline (preprocessor + model) used by the Streamlit app.
- `models/metrics.csv` — CSV log of evaluation metrics for candidate models evaluated during training. Columns include `model`, `rmse`, `r2`, and `run_time_utc`.

You can inspect `models/metrics.csv` to compare models across training runs.

Quick start
1. Create and activate a Python environment (Python 3.8+ recommended).
2. Install dependencies:

   pip install -r requirements.txt

3. Train using sample data (this will save a model to `models/car_price_model.pkl`):

   python train.py

4. Run the Streamlit app:

   streamlit run app.py

Using your own dataset
 - Place a CSV at `data/cars.csv`. The script expects columns: `brand,model,year,km_driven,mileage,engine,max_power,owner,fuel_type,transmission,seller_type,selling_price`.
 - If your dataset has different column names, adapt `train.py` accordingly.

Notes
 - This project includes a minimal sample dataset so you can try the app end-to-end offline. For production-quality models, use a larger Indian used-car dataset (e.g., CarDekho dataset from Kaggle).

Notes on preprocessing
----------------------

Run `python data_processing.py` to create a cleaned dataset at `data/cleaned_cars.csv`. The cleaning script attempts to coerce numeric fields, normalize owner descriptions like `First Owner`, and remove obvious unit suffixes (e.g., `bhp`, `cc`). If you provide your own dataset, ensure it contains the expected columns described below.

Developed by Somen Parida | Data Science Ecosystem Project
