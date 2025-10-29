"""
Data loading and preprocessing utilities for the used-car project.

Creates a cleaned CSV at `data/cleaned_cars.csv` when run as a script.
Functions are intentionally small and testable.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "cars.csv"
DATA_SAMPLE = ROOT / "data" / "sample_cars.csv"
DATA_CLEAN = ROOT / "data" / "cleaned_cars.csv"


def load_raw(path: Path | str = None) -> pd.DataFrame:
    p = Path(path) if path else (DATA_RAW if DATA_RAW.exists() else DATA_SAMPLE)
    df = pd.read_csv(p)
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and feature engineering.

    - Coerce numeric columns
    - Compute `age` from `year`
    - Normalize textual fields (strip)
    - Drop rows with missing target (`selling_price`) or too many NAs
    """
    df = df.copy()

    # strip string columns
    for c in df.select_dtypes(include=[object]).columns:
        df[c] = df[c].astype(str).str.strip()

    # normalize common owner strings (e.g., 'First Owner', 'Second Owner')
    def parse_owner(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        # common textual patterns
        if s.isdigit():
            return int(s)
        if "first" in s:
            return 1
        if "second" in s:
            return 2
        if "third" in s:
            return 3
        if "fourth" in s or "four+" in s or "fourth & above" in s:
            return 4
        # fallback: try to extract digits
        digits = ''.join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else np.nan

    # numeric coercion: handle columns that may contain units like 'bhp', 'kmpl', 'cc'
    numeric_cols = ["year", "km_driven", "mileage", "engine", "max_power", "owner", "selling_price"]
    for c in numeric_cols:
        if c in df.columns:
            # try to extract numeric part first
            # keep decimals and digits
            df[c] = df[c].astype(str).str.replace(r"[^0-9\.]+", "", regex=True)
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # if owner column contains textual descriptions, normalize
    if "owner" in df.columns:
        try:
            # if owner is not fully numeric, map common words
            mask_non_numeric = df["owner"].apply(lambda v: not str(v).strip().isdigit())
            if mask_non_numeric.any():
                df.loc[mask_non_numeric, "owner"] = df.loc[mask_non_numeric, "owner"].apply(parse_owner)
        except Exception:
            # fallback: ignore
            pass

    # compute age
    if "year" in df.columns:
        df["age"] = pd.Timestamp.now().year - df["year"]

    # sensible sanity: if mileage or max_power are zero or extreme, set NaN
    if "mileage" in df.columns:
        df.loc[df["mileage"] <= 0, "mileage"] = np.nan
        df.loc[df["mileage"] > 200, "mileage"] = np.nan
    if "max_power" in df.columns:
        df.loc[df["max_power"] <= 0, "max_power"] = np.nan
        df.loc[df["max_power"] > 2000, "max_power"] = np.nan

    # drop rows with missing selling_price when present
    if "selling_price" in df.columns:
        df = df.dropna(subset=["selling_price"])

    # drop any rows with remaining NAs (small sample)
    df = df.dropna()

    # cast types for consistency
    if "owner" in df.columns:
        df["owner"] = df["owner"].astype(int)

    return df


def save_cleaned(df: pd.DataFrame, path: Path | str = None):
    p = Path(path) if path else DATA_CLEAN
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def main():
    df = load_raw()
    dfc = clean_df(df)
    out = save_cleaned(dfc)
    print(f"Saved cleaned data to {out} ({len(dfc)} rows)")


if __name__ == "__main__":
    main()
