from __future__ import annotations
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import load_data_csv, compute_targets, build_preprocessor
from .schema import NUM_COLS, CAT_COLS, TARGET_COL

def train(data_path: str, out_dir: str) -> dict:
    df = load_data_csv(data_path)
    # basic schema check
    missing = sorted(set(NUM_COLS + CAT_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = compute_targets(df)
    X = df[NUM_COLS + CAT_COLS].copy()
    y = df[TARGET_COL].values

    # --- robust split that works for small datasets with stratify ---
    strat = df["gender"] if "gender" in df and df["gender"].nunique() >= 2 else None
    n = len(X)
    base_float = 0.2
    base_int = int(np.ceil(base_float * n))
    if strat is not None:
        n_classes = df["gender"].nunique()
        test_n = max(base_int, n_classes)               # ensure >= 1 per class in test
        test_n = min(test_n, n - 1)                     # leave at least 1 sample for train
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_n, random_state=42, stratify=strat
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=base_float, random_state=42, stratify=None
        )


    pre = build_preprocessor(categories=None)
    model = LinearRegression()

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)
    mae = float(mean_absolute_error(y_val, preds))
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))

    # counterfactual adjusted gap by swapping gender (binary heuristic)
    if "gender" in X_val.columns:
        X_cf = X_val.copy()
        mapping = {"Male":"Female", "Female":"Male", "M":"F", "F":"M"}
        X_cf["gender"] = X_cf["gender"].map(lambda g: mapping.get(g, g))
        adj_gap = float(np.mean(pipe.predict(X_val) - pipe.predict(X_cf)))
    else:
        adj_gap = float("nan")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, Path(out_dir)/"model.pkl")

    metrics = {"mae": mae, "rmse": rmse, "adjusted_gap_log": adj_gap}
    (Path(out_dir)/"metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV with required columns")
    ap.add_argument("--out", default="artifacts/", help="Output dir for model + metrics")
    args = ap.parse_args()
    metrics = train(args.data, args.out)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
