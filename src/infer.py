from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from .schema import NUM_COLS, CAT_COLS

def predict(model_path: str, data_path: str, out_path: str):
    pipe = joblib.load(model_path)
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]
    X = df[NUM_COLS + CAT_COLS].copy()
    preds = pipe.predict(X)
    df_out = df.copy()
    df_out["pred_log_total_comp"] = preds
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/model.pkl")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="predictions/preds.csv")
    args = ap.parse_args()
    out = predict(args.model, args.data, args.out)
    print(f"Wrote predictions to {out}")

if __name__ == "__main__":
    main()
