from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .schema import NUM_COLS, CAT_COLS

def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_comp"] = df["basepay"].astype(float) + df["bonus"].astype(float)
    df["log_total_comp"] = np.log(df["total_comp"].clip(lower=1.0))
    return df

def build_preprocessor(categories: dict[str, list[str]] | None = None) -> ColumnTransformer:
    # Build a ColumnTransformer that scales numeric cols and one-hot encodes categoricals.
    # If a categories dict is provided, use fixed category lists for stable train/infer parity.
    cat = OneHotEncoder(handle_unknown="ignore", categories=None if categories is None else [categories[c] for c in CAT_COLS])
    num = StandardScaler()
    pre = ColumnTransformer(
        transformers=[
            ("num", num, NUM_COLS),
            ("cat", cat, CAT_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre

def load_data_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    return df
