import pandas as pd
from pathlib import Path
from data_processing import load_raw, clean_df


def test_clean_sample_has_expected_columns():
    df = load_raw()
    dfc = clean_df(df)
    # required columns present
    for col in ["brand", "model", "age", "km_driven", "mileage", "engine", "max_power", "owner"]:
        assert col in dfc.columns


def test_numeric_types_after_cleaning():
    df = load_raw()
    dfc = clean_df(df)
    assert pd.api.types.is_integer_dtype(dfc["owner"]) or pd.api.types.is_numeric_dtype(dfc["owner"])
    for c in ["age", "km_driven", "mileage", "engine", "max_power"]:
        assert pd.api.types.is_numeric_dtype(dfc[c])
