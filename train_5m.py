"""Training pipeline for training_5m.csv — 5-minute regime classification (baseline).

6 classes: RANGE, TREND_UP, TREND_DOWN, STRONG_UP, STRONG_DOWN, Warmup→dropped.
Supports --mode static|optuna|both for hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from train_common import (
    run_optuna_study, train_ensemble, evaluate, export_models,
    chronological_split, STATIC_XGB_PARAMS, STATIC_LGBM_PARAMS,
)

REGIME_CLASSES = [
    "RANGE",        # id 0
    "TREND_UP",     # id 1
    "TREND_DOWN",   # id 2
    "STRONG_UP",    # id 3
    "STRONG_DOWN",  # id 4
]

TARGET_COL = "target_regime_id"

NON_FEATURE_COLS = {
    TARGET_COL,
    "bucket_ts",
    "symbol",
    "timeframe",
    "target_regime",
    "trend_regime",
    "trend_regime_id",
    "vol_regime",
    "vol_regime_id",
    "liq_regime",
    "liq_regime_id",
}


def load_data(csv_path: str):
    """Load CSV, extract features and target, drop Warmup rows."""
    df = pd.read_csv(csv_path)

    if "bucket_ts" in df.columns:
        df = df.sort_values("bucket_ts").reset_index(drop=True)

    warmup_mask = df[TARGET_COL] == 5
    if warmup_mask.any():
        print(f"  Dropping {warmup_mask.sum()} Warmup rows")
        df = df[~warmup_mask].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    feature_names = feature_cols

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    y = df[TARGET_COL].astype(int)

    valid_mask = y.between(0, 4)
    if (~valid_mask).any():
        print(f"  Dropping {(~valid_mask).sum()} rows with unexpected target values")
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

    print(f"  {len(X)} samples, {X.shape[1]} features")
    print(f"  Class distribution:")
    for cls in sorted(y.unique()):
        count = (y == cls).sum()
        print(f"    {REGIME_CLASSES[cls]:12s} (id={cls}): {count:>6d} ({count/len(y)*100:.1f}%)")

    return X, y.astype(int), feature_names


def run_pipeline(csv_path, output_dir, mode, n_trials, xgb_params, lgbm_params):
    """Run train + evaluate + export for one mode."""
    num_class = len(REGIME_CLASSES)

    X, y, feature_names = load_data(csv_path)

    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)
    print(f"\n  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    if xgb_params is None and lgbm_params is None:
        print(f"\n  Using static hyperparameters")
    else:
        print(f"\n  Using Optuna-tuned hyperparameters")

    xgb_model, lgbm_model = train_ensemble(
        X_train, y_train, X_val, y_val,
        num_class=num_class, xgb_params=xgb_params, lgbm_params=lgbm_params,
    )
    acc, f1 = evaluate(xgb_model, lgbm_model, X_test, y_test, REGIME_CLASSES, tag="Test ")

    print(f"\n  Retraining on full data for export...")
    xgb_full, lgbm_full = train_ensemble(
        X, y, num_class=num_class, xgb_params=xgb_params, lgbm_params=lgbm_params,
    )
    export_models(xgb_full, lgbm_full, feature_names, output_dir, "5m", REGIME_CLASSES)

    return acc, f1


def main(csv_path="training_5m.csv", output_dir="trained_output_5m",
         mode="static", n_trials=100):
    num_class = len(REGIME_CLASSES)

    if mode in ("static", "both"):
        print("\n" + "=" * 60)
        print("5m Training — STATIC hyperparameters")
        print("=" * 60)
        acc_static, f1_static = run_pipeline(
            csv_path, output_dir, mode, n_trials,
            STATIC_XGB_PARAMS, STATIC_LGBM_PARAMS,
        )

    if mode in ("optuna", "both"):
        print("\n" + "=" * 60)
        print("5m Training — OPTUNA hyperparameters")
        print("=" * 60)

        # Load data once for Optuna study
        X, y, feature_names = load_data(csv_path)
        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)

        print(f"\n  Running Optuna ({n_trials} trials)...")
        xgb_best, lgbm_best = run_optuna_study(
            X_train, y_train, X_val, y_val, num_class, n_trials=n_trials,
        )

        optuna_dir = output_dir + "_optuna"
        acc_optuna, f1_optuna = run_pipeline(
            csv_path, optuna_dir, mode, n_trials,
            xgb_best, lgbm_best,
        )

    if mode == "both":
        print("\n" + "=" * 60)
        print("COMPARISON: Static vs Optuna")
        print("=" * 60)
        print(f"  Static  accuracy: {acc_static:.4f}  |  macro-F1: {f1_static:.4f}")
        print(f"  Optuna  accuracy: {acc_optuna:.4f}  |  macro-F1: {f1_optuna:.4f}")
        print(f"  Delta   accuracy: {acc_optuna - acc_static:+.4f}  |  macro-F1: {f1_optuna - f1_static:+.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 5m regime classifier")
    parser.add_argument("--csv", default="training_5m.csv")
    parser.add_argument("--output", default="trained_output_5m")
    parser.add_argument("--mode", choices=["static", "optuna", "both"], default="static")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trial count")
    args = parser.parse_args()
    main(args.csv, args.output, args.mode, args.trials)
