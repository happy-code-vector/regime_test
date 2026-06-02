"""5m training — all improvements combined.

Stratified classes + class weights + feature engineering.
Supports --mode static|optuna|both.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from train_common import (
    run_optuna_study, train_ensemble, evaluate, export_models,
    chronological_split, stratify_classes, compute_sample_weights,
    add_engineered_features, STATIC_XGB_PARAMS, STATIC_LGBM_PARAMS,
)

REGIME_CLASSES = [
    "RANGE", "TREND_UP", "TREND_DOWN", "STRONG_UP", "STRONG_DOWN",
]

TARGET_COL = "target_regime_id"

NON_FEATURE_COLS = {
    TARGET_COL, "bucket_ts", "symbol", "timeframe",
    "target_regime", "trend_regime", "trend_regime_id",
    "vol_regime", "vol_regime_id", "liq_regime", "liq_regime_id",
}

CORE_FEATURES = [
    "slope_z_5m", "adx_z_5m",
    "slope_z_15m", "adx_z_15m",
    "slope_z_30m", "adx_z_30m",
    "slope_z_1h", "adx_z_1h",
    "realized_vol_z_5m", "atr_z_5m",
]


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if "bucket_ts" in df.columns:
        df = df.sort_values("bucket_ts").reset_index(drop=True)

    warmup_mask = df[TARGET_COL] == 5
    if warmup_mask.any():
        print(f"  Dropping {warmup_mask.sum()} Warmup rows")
        df = df[~warmup_mask].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    y = df[TARGET_COL].astype(int)

    valid_mask = y.between(0, 4)
    if (~valid_mask).any():
        X_raw = X_raw[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

    X = add_engineered_features(X_raw, CORE_FEATURES)
    feature_names = list(X.columns)

    print(f"  {len(X)} samples, {X.shape[1]} features")
    for cls in sorted(y.unique()):
        count = (y == cls).sum()
        print(f"    {REGIME_CLASSES[cls]:12s}: {count:>6d} ({count/len(y)*100:.1f}%)")

    return X, y.astype(int), feature_names


def run_pipeline(csv_path, output_dir, mode, n_trials, xgb_params, lgbm_params):
    num_class = len(REGIME_CLASSES)
    X, y, feature_names = load_data(csv_path)

    X_bal, y_bal = stratify_classes(X, y, min_samples=1000, max_ratio=5.0, regime_classes=REGIME_CLASSES)

    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X_bal, y_bal)
    print(f"\n  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    print(f"\n  Computing class weights from training set...")
    sw_train = compute_sample_weights(y_train)

    if xgb_params is None:
        print(f"\n  Using static hyperparameters")
    else:
        print(f"\n  Using Optuna-tuned hyperparameters")

    xgb_model, lgbm_model = train_ensemble(
        X_train, y_train, X_val, y_val,
        num_class=num_class, xgb_params=xgb_params, lgbm_params=lgbm_params,
        sample_weight=sw_train,
    )
    acc = evaluate(xgb_model, lgbm_model, X_test, y_test, REGIME_CLASSES, tag="Test ")

    print(f"\n  Retraining on full balanced data for export...")
    sw_full = compute_sample_weights(y_bal)
    xgb_full, lgbm_full = train_ensemble(
        X_bal, y_bal, num_class=num_class, xgb_params=xgb_params, lgbm_params=lgbm_params,
        sample_weight=sw_full,
    )
    export_models(xgb_full, lgbm_full, feature_names, output_dir, "5m", REGIME_CLASSES)
    return acc


def main(csv_path="training_5m.csv", output_dir="trained_output_5m_improved",
         mode="static", n_trials=50):
    num_class = len(REGIME_CLASSES)

    if mode in ("static", "both"):
        print("\n" + "=" * 60)
        print("5m Improved — STATIC hyperparameters")
        print("=" * 60)
        acc_static = run_pipeline(csv_path, output_dir, mode, n_trials, STATIC_XGB_PARAMS, STATIC_LGBM_PARAMS)

    if mode in ("optuna", "both"):
        print("\n" + "=" * 60)
        print("5m Improved — OPTUNA hyperparameters")
        print("=" * 60)
        X, y, feature_names = load_data(csv_path)
        X_bal, y_bal = stratify_classes(X, y, min_samples=1000, max_ratio=5.0, regime_classes=REGIME_CLASSES)
        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X_bal, y_bal)
        sw_train = compute_sample_weights(y_train)

        print(f"\n  Running Optuna ({n_trials} trials)...")
        xgb_best, lgbm_best = run_optuna_study(
            X_train, y_train, X_val, y_val, num_class,
            sample_weight=sw_train, n_trials=n_trials,
        )

        acc_optuna = run_pipeline(csv_path, output_dir + "_optuna", mode, n_trials, xgb_best, lgbm_best)

    if mode == "both":
        print("\n" + "=" * 60)
        print("COMPARISON: Static vs Optuna")
        print("=" * 60)
        print(f"  Static  accuracy: {acc_static:.4f}")
        print(f"  Optuna  accuracy: {acc_optuna:.4f}")
        print(f"  Delta:            {acc_optuna - acc_static:+.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="5m training — all improvements")
    parser.add_argument("--csv", default="training_5m.csv")
    parser.add_argument("--output", default="trained_output_5m_improved")
    parser.add_argument("--mode", choices=["static", "optuna", "both"], default="static")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    main(args.csv, args.output, args.mode, args.trials)
