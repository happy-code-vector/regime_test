"""Training pipeline for training_15m.csv — single-timeframe regime classification.

Differences from train.py:
  - Flat feature names (no timeframe prefix filtering)
  - Target column is target_regime_id (integer 0-9)
  - Regime IDs 7/8/9 are extremely rare → merged into class 6 ("OTHER")
  - Uses bucket_ts for chronological ordering in splits
  - Single timeframe (15m) only, no multi-TF or combined model
"""

import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgbm
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Original 7 classes + OTHER for rare IDs (7/8/9)
REGIME_CLASSES = [
    "REGIME_0",   # id 0
    "REGIME_1",   # id 1
    "REGIME_2",   # id 2
    "REGIME_3",   # id 3
    "REGIME_4",   # id 4
    "REGIME_5",   # id 5
    "REGIME_6",   # id 6
    "OTHER",      # ids 7/8/9 merged
]

TARGET_COL = "target_regime_id"

# Columns that are NOT features
NON_FEATURE_COLS = {TARGET_COL, "bucket_ts"}


def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load CSV, extract features and target, merge rare classes."""
    df = pd.read_csv(csv_path)

    # Sort by timestamp to preserve chronological order
    if "bucket_ts" in df.columns:
        df = df.sort_values("bucket_ts").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    feature_names = feature_cols

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    y_raw = df[TARGET_COL].astype(int)

    # Merge rare classes 7/8/9 → 7 ("OTHER")
    y = y_raw.copy()
    y[y >= 7] = 7

    # Drop any rows with target outside 0-7
    valid_mask = y.between(0, 7)
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


def chronological_split(X, y, train_ratio=0.70, val_ratio=0.15):
    """Simple chronological split — data is already sorted by bucket_ts."""
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = (
        X.iloc[:train_end].reset_index(drop=True),
        X.iloc[train_end:val_end].reset_index(drop=True),
        X.iloc[val_end:].reset_index(drop=True),
        y.iloc[:train_end].reset_index(drop=True),
        y.iloc[train_end:val_end].reset_index(drop=True),
        y.iloc[val_end:].reset_index(drop=True),
    )

    for name, y_part in [("train", splits[3]), ("val", splits[4]), ("test", splits[5])]:
        print(f"    {name}: {len(y_part)} samples")

    return splits


def stratify_classes(X, y, min_samples=500, max_ratio=10.0):
    """Downsample dominant classes and oversample rare classes."""
    counts = y.value_counts()
    max_keep = int(counts.max() / max_ratio)
    target_per_class = max(max_keep, min_samples)

    indices = []
    for cls in sorted(counts.index):
        cls_idx = list(y[y == cls].index)
        n_have = len(cls_idx)

        if n_have >= target_per_class:
            sampled = np.random.choice(cls_idx, target_per_class, replace=False)
            indices.extend(sampled)
        else:
            indices.extend(cls_idx)
            extra = np.random.choice(cls_idx, target_per_class - n_have, replace=True)
            indices.extend(extra)

    np.random.shuffle(indices)
    print(f"  Stratified: {len(y)} -> {len(indices)} samples (target {target_per_class}/class)")
    for cls in sorted(counts.index):
        print(f"    {REGIME_CLASSES[cls]:12s}: {(y.iloc[indices] == cls).sum()}")
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)


def train_ensemble(X_train, y_train, X_val=None, y_val=None):
    """Train XGBoost + LightGBM ensemble."""
    num_class = len(REGIME_CLASSES)

    print(f"\n  Training XGBoost on {len(X_train)} samples, {X_train.shape[1]} features")
    start = time.time()

    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=25 if X_val is not None else None,
        verbosity=0,
    )
    fit_kw = {}
    if X_val is not None:
        fit_kw["eval_set"] = [(X_val.values, y_val.values)]
        fit_kw["verbose"] = False
    xgb_model.fit(X_train.values, y_train.values, **fit_kw)
    print(f"  XGBoost trained in {time.time() - start:.1f}s")

    print(f"  Training LightGBM on {len(X_train)} samples")
    start = time.time()

    lgbm_model = lgbm.LGBMClassifier(
        objective="multiclass",
        num_class=num_class,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )
    lgbm_kw = {}
    if X_val is not None:
        lgbm_kw["eval_set"] = [(X_val.values, y_val.values)]
    lgbm_model.fit(X_train.values, y_train.values, **lgbm_kw)
    print(f"  LightGBM trained in {time.time() - start:.1f}s")

    return xgb_model, lgbm_model


def evaluate(xgb_model, lgbm_model, X_test, y_test):
    """Evaluate ensemble and print metrics."""
    proba_xgb = xgb_model.predict_proba(X_test.values)
    proba_lgbm = lgbm_model.predict_proba(X_test.values)
    proba_ensemble = (proba_xgb + proba_lgbm) / 2
    pred_ensemble = proba_ensemble.argmax(axis=1)

    acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test.values))
    acc_lgbm = accuracy_score(y_test, lgbm_model.predict(X_test.values))
    acc_ens = accuracy_score(y_test, pred_ensemble)

    print(f"\n  XGBoost accuracy:  {acc_xgb:.4f}")
    print(f"  LightGBM accuracy: {acc_lgbm:.4f}")
    print(f"  Ensemble accuracy: {acc_ens:.4f}")

    # Only show labels/classes that appear in y_test
    present_labels = sorted(y_test.unique())
    present_names = [REGIME_CLASSES[i] for i in present_labels]
    print(classification_report(
        y_test, pred_ensemble,
        labels=present_labels, target_names=present_names,
        zero_division=0,
    ))

    return acc_ens


def export_models(xgb_model, lgbm_model, feature_names, output_dir):
    """Save models + metadata."""
    out = Path(output_dir) / "15m"
    out.mkdir(parents=True, exist_ok=True)

    xgb_path = out / "xgb_regime.pkl"
    lgbm_path = out / "lgbm_regime.pkl"
    meta_path = out / "metadata.json"
    features_path = out / "feature_names.json"

    joblib.dump(xgb_model, xgb_path)
    joblib.dump(lgbm_model, lgbm_path)
    json.dump(feature_names, open(features_path, "w"), indent=2)
    json.dump({
        "timeframe": "15m",
        "regime_classes": REGIME_CLASSES,
        "model_version": f"v1_{int(time.time())}",
        "xgb_file": xgb_path.name,
        "lgbm_file": lgbm_path.name,
        "feature_file": features_path.name,
        "num_features": len(feature_names),
        "target_column": TARGET_COL,
        "class_mapping": {
            "0": "REGIME_0", "1": "REGIME_1", "2": "REGIME_2",
            "3": "REGIME_3", "4": "REGIME_4", "5": "REGIME_5",
            "6": "REGIME_6", "7": "OTHER", "8": "OTHER", "9": "OTHER",
        },
    }, open(meta_path, "w"), indent=2)

    print(f"\n  Exported to {out.resolve()}\\")
    print(f"    {xgb_path.name}  ({xgb_path.stat().st_size / 1024:.0f} KB)")
    print(f"    {lgbm_path.name}  ({lgbm_path.stat().st_size / 1024:.0f} KB)")


def main(csv_path="training_15m.csv", output_dir="trained_output_15m"):
    print("=" * 60)
    print("Regime Classification — 15m Single-Timeframe Training")
    print("=" * 60)

    X, y, feature_names = load_data(csv_path)

    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)
    print(f"\n  Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    xgb_model, lgbm_model = train_ensemble(X_train, y_train, X_val, y_val)
    evaluate(xgb_model, lgbm_model, X_test, y_test)

    print(f"\n  Retraining on full data for export...")
    xgb_full, lgbm_full = train_ensemble(X, y)
    export_models(xgb_full, lgbm_full, feature_names, output_dir)

    print(f"\n{'=' * 60}")
    print(f"Done. Output: {Path(output_dir).resolve()}\\")
    print(f"  15m/ — XGBoost + LightGBM ensemble")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train regime classifier on training_15m.csv")
    parser.add_argument("--csv", default="training_15m.csv", help="Path to training CSV")
    parser.add_argument("--output", default="trained_output_15m", help="Output directory")
    args = parser.parse_args()
    main(args.csv, args.output)
