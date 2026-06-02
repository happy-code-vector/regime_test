"""15m training — 3-class merge (no smoothing).

8 GMM classes merged to BULL/RANGE/BEAR.
Supports --mode static|optuna|both.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from train_common import (
    run_optuna_study, train_ensemble, evaluate, export_models,
    chronological_split, STATIC_XGB_PARAMS, STATIC_LGBM_PARAMS,
)

REGIME_CLASSES = ["BULL", "RANGE", "BEAR"]

CLASS_MAP_3 = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2}

TARGET_COL = "target_regime_id"
NON_FEATURE_COLS = {TARGET_COL, "bucket_ts"}


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if "bucket_ts" in df.columns:
        df = df.sort_values("bucket_ts").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    y_raw = df[TARGET_COL].astype(int).clip(upper=7)
    print(f"\n  Merging 8 classes -> 3 ({', '.join(REGIME_CLASSES)})")
    y = y_raw.map(CLASS_MAP_3)

    if y.isnull().any():
        mask = y.notnull()
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

    y = y.astype(int)
    print(f"\n  {len(X)} samples, {X.shape[1]} features, {len(REGIME_CLASSES)} classes")
    for cls in range(len(REGIME_CLASSES)):
        count = (y == cls).sum()
        print(f"    {REGIME_CLASSES[cls]:6s}: {count:>6d} ({count/len(y)*100:.1f}%)")

    return X, y, list(X.columns)


def run_pipeline(csv_path, output_dir, mode, n_trials, xgb_params, lgbm_params):
    num_class = len(REGIME_CLASSES)
    X, y, feature_names = load_data(csv_path)

    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)
    print(f"\n  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    if xgb_params is None:
        print(f"\n  Using static hyperparameters")
    else:
        print(f"\n  Using Optuna-tuned hyperparameters")

    xgb_model, lgbm_model = train_ensemble(
        X_train, y_train, X_val, y_val,
        num_class=num_class, xgb_params=xgb_params, lgbm_params=lgbm_params,
    )
    acc = evaluate(xgb_model, lgbm_model, X_test, y_test, REGIME_CLASSES, tag="Test ")

    print(f"\n  Retraining on full data for export...")
    xgb_full, lgbm_full = train_ensemble(
        X, y, num_class=num_class, xgb_params=xgb_params, lgbm_params=lgbm_params,
    )
    export_models(xgb_full, lgbm_full, feature_names, output_dir, "15m", REGIME_CLASSES,
                  metadata_extra={"approach": "merge_3class", "class_mapping": CLASS_MAP_3})
    return acc


def main(csv_path="training_15m.csv", output_dir="trained_output_15m_merge",
         mode="static", n_trials=100):
    num_class = len(REGIME_CLASSES)

    if mode in ("static", "both"):
        print("\n" + "=" * 60)
        print("15m Merge — STATIC hyperparameters")
        print("=" * 60)
        acc_static = run_pipeline(csv_path, output_dir, mode, n_trials, STATIC_XGB_PARAMS, STATIC_LGBM_PARAMS)

    if mode in ("optuna", "both"):
        print("\n" + "=" * 60)
        print("15m Merge — OPTUNA hyperparameters")
        print("=" * 60)
        X, y, feature_names = load_data(csv_path)
        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)

        print(f"\n  Running Optuna ({n_trials} trials)...")
        xgb_best, lgbm_best = run_optuna_study(X_train, y_train, X_val, y_val, num_class, n_trials=n_trials)

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
    parser = argparse.ArgumentParser(description="15m training — 3-class merge")
    parser.add_argument("--csv", default="training_15m.csv")
    parser.add_argument("--output", default="trained_output_15m_merge")
    parser.add_argument("--mode", choices=["static", "optuna", "both"], default="static")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    main(args.csv, args.output, args.mode, args.trials)
