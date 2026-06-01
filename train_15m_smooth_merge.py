"""Approach 1+2 COMBINED: Smoothing + class merge.

1. Rolling majority vote (window=7) stabilizes noisy GMM labels.
2. 8 classes collapse to 3 (BULL / RANGE / BEAR) based on slope direction.
"""

import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgbm
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

TARGET_COL = "target_regime_id"
NON_FEATURE_COLS = {TARGET_COL, "bucket_ts"}

REGIME_CLASSES = ["BULL", "RANGE", "BEAR"]

CLASS_MAP_3 = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2}


def smooth_labels(y: pd.Series, window: int = 7) -> pd.Series:
    """Rolling majority vote to stabilize noisy GMM labels."""
    raw_autocorr = (y.values[1:] == y.values[:-1]).mean()
    print(f"  Raw lag-1 autocorrelation: {raw_autocorr*100:.1f}%")

    smoothed = (
        y.rolling(window, center=True, min_periods=1)
        .apply(lambda x: pd.Series(x).mode().iloc[0])
        .astype(int)
    )

    new_autocorr = (smoothed.values[1:] == smoothed.values[:-1]).mean()
    print(f"  Smoothed (window={window}) lag-1 autocorrelation: {new_autocorr*100:.1f}%")
    return smoothed


def load_data(csv_path: str, smooth_window: int = 7) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load CSV, smooth labels, merge to 3 classes."""
    df = pd.read_csv(csv_path)

    if "bucket_ts" in df.columns:
        df = df.sort_values("bucket_ts").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    feature_names = feature_cols

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    # Step 1: smooth raw labels
    y_raw = df[TARGET_COL].astype(int).clip(upper=7)
    print(f"\n  Step 1: Smoothing labels (window={smooth_window})")
    y_smooth = smooth_labels(y_raw, window=smooth_window)

    # Step 2: merge to 3 classes
    print(f"\n  Step 2: Merging 8 classes -> 3 ({', '.join(REGIME_CLASSES)})")
    y = y_smooth.map(CLASS_MAP_3)

    if y.isnull().any():
        bad = y.isnull().sum()
        print(f"  Dropping {bad} rows with unmapped labels")
        mask = y.notnull()
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

    y = y.astype(int)

    print(f"\n  Final: {len(X)} samples, {X.shape[1]} features, {len(REGIME_CLASSES)} classes")
    for cls in range(len(REGIME_CLASSES)):
        count = (y == cls).sum()
        print(f"    {REGIME_CLASSES[cls]:6s}: {count:>6d} ({count/len(y)*100:.1f}%)")

    return X, y, feature_names


def chronological_split(X, y, train_ratio=0.70, val_ratio=0.15):
    """Simple chronological split."""
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
        counts = y_part.value_counts().sort_index()
        dist = "  ".join([f"{REGIME_CLASSES[c]}={n}" for c, n in counts.items()])
        print(f"    {name}: {len(y_part)} samples  ({dist})")

    return splits


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


def evaluate(xgb_model, lgbm_model, X_test, y_test, tag=""):
    """Evaluate ensemble and print metrics."""
    proba_xgb = xgb_model.predict_proba(X_test.values)
    proba_lgbm = lgbm_model.predict_proba(X_test.values)
    proba_ensemble = (proba_xgb + proba_lgbm) / 2
    pred_ensemble = proba_ensemble.argmax(axis=1)

    acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test.values))
    acc_lgbm = accuracy_score(y_test, lgbm_model.predict(X_test.values))
    acc_ens = accuracy_score(y_test, pred_ensemble)

    print(f"\n  {tag}XGBoost accuracy:  {acc_xgb:.4f}")
    print(f"  {tag}LightGBM accuracy: {acc_lgbm:.4f}")
    print(f"  {tag}Ensemble accuracy: {acc_ens:.4f}")

    present_labels = sorted(y_test.unique())
    present_names = [REGIME_CLASSES[i] for i in present_labels]
    print(classification_report(
        y_test, pred_ensemble,
        labels=present_labels, target_names=present_names,
        zero_division=0,
    ))

    cm = confusion_matrix(y_test, pred_ensemble, labels=range(len(REGIME_CLASSES)))
    print(f"  {tag}Confusion matrix (rows=true, cols=pred):")
    header = "       " + "  ".join([f"{n:>6s}" for n in REGIME_CLASSES])
    print(header)
    for i in range(len(REGIME_CLASSES)):
        row = "  ".join([f"{cm[i,j]:>6d}" for j in range(len(REGIME_CLASSES))])
        print(f"  {REGIME_CLASSES[i]:5s}: {row}")

    return acc_ens


def export_models(xgb_model, lgbm_model, feature_names, output_dir, smooth_window):
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
        "approach": "smooth_merge",
        "regime_classes": REGIME_CLASSES,
        "model_version": f"v2_smooth_merge_{int(time.time())}",
        "xgb_file": xgb_path.name,
        "lgbm_file": lgbm_path.name,
        "feature_file": features_path.name,
        "num_features": len(feature_names),
        "target_column": TARGET_COL,
        "smooth_window": smooth_window,
        "class_mapping": {
            "raw_to_merged": CLASS_MAP_3,
            "description": "8 GMM classes smoothed (window=7) then merged to 3: BULL(0,1,2), RANGE(3), BEAR(4,5,6,7)",
        },
    }, open(meta_path, "w"), indent=2)

    print(f"\n  Exported to {out.resolve()}\\")
    print(f"    {xgb_path.name}  ({xgb_path.stat().st_size / 1024:.0f} KB)")
    print(f"    {lgbm_path.name}  ({lgbm_path.stat().st_size / 1024:.0f} KB)")


def main(csv_path="training_15m.csv", output_dir="trained_output_15m_smooth_merge", smooth_window=7):
    print("=" * 60)
    print("Approach 1+2: Smooth + Merge COMBINED")
    print("=" * 60)

    X, y, feature_names = load_data(csv_path, smooth_window=smooth_window)

    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)
    print(f"\n  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    xgb_model, lgbm_model = train_ensemble(X_train, y_train, X_val, y_val)
    evaluate(xgb_model, lgbm_model, X_test, y_test, tag="Test ")

    print(f"\n  Retraining on full data for export...")
    xgb_full, lgbm_full = train_ensemble(X, y)
    export_models(xgb_full, lgbm_full, feature_names, output_dir, smooth_window)

    print(f"\n{'=' * 60}")
    print(f"Done. Output: {Path(output_dir).resolve()}\\")
    print(f"  15m/ — XGBoost + LightGBM ensemble (3-class, smoothed)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Approach 1+2: Smooth + Merge")
    parser.add_argument("--csv", default="training_15m.csv")
    parser.add_argument("--output", default="trained_output_15m_smooth_merge")
    parser.add_argument("--smooth-window", type=int, default=7)
    args = parser.parse_args()
    main(args.csv, args.output, args.smooth_window)
