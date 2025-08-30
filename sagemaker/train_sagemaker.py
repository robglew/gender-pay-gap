import argparse, os, json
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

from src.features import compute_targets, build_preprocessor
from src.schema import NUM_COLS, CAT_COLS, TARGET_COL

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/training"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = parser.parse_args()

    # Expect a single CSV in the training channel
    train_files = list(Path(args.train).glob("*.csv"))
    if not train_files:
        raise FileNotFoundError(f"No CSV found in {args.train}")
    df = pd.read_csv(train_files[0])
    df.columns = [c.strip().lower() for c in df.columns]

    missing = sorted(set(NUM_COLS + CAT_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Prepare targets
    df = compute_targets(df)
    X = df[NUM_COLS + CAT_COLS].copy()
    y = df[TARGET_COL].values

    pre = build_preprocessor(categories=None)
    model = LinearRegression()
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X, y)

    # Evaluate in-sample performance
    preds = pipe.predict(X)
    mae = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))

    # Save model + metrics
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, Path(args.model_dir) / "model.pkl")
    Path(args.model_dir, "metrics.json").write_text(json.dumps({"mae": mae, "rmse": rmse}, indent=2))

if __name__ == "__main__":
    main()
