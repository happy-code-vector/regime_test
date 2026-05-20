"""Training pipeline — trains XGBoost + LightGBM per timeframe + combined model."""

import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgbm
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import joblib

REGIME_CLASSES = [
    "LIQ_CASCADE",
    "VOL_EXPAND",
    "TRANSITION",
    "TREND_BEAR",
    "TREND_BULL",
    "VOL_COMPRESS",
    "RANGE",
]

ALL_LABEL_COLUMNS = {
    "15m_regime_state", "15m_regime", "15m_regime_confidence", "15m_confidence_liq",
    "future_15m_regime_state", "future_15m_regime", "has_label_15m",
    "1h_regime_state", "1h_regime", "1h_regime_confidence", "1h_confidence_liq",
    "future_1h_regime_state", "future_1h_regime", "has_label_1h",
}

TIMEFRAMES = {
    "15m": {"prefix": "15m_", "target": "15m_regime"},
    "1h":  {"prefix": "1h_",  "target": "1h_regime"},
}


def load_for_timeframe(csv_path: str, tf: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load CSV and extract only the features + label for one timeframe."""
    df = pd.read_csv(csv_path)
    cfg = TIMEFRAMES[tf]
    prefix = cfg["prefix"]
    target_col = cfg["target"]

    feature_cols = [c for c in df.columns if c.startswith(prefix) and c not in ALL_LABEL_COLUMNS]
    feature_names = feature_cols

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    y = df[target_col].map({name: i for i, name in enumerate(REGIME_CLASSES)})

    if y.isnull().any():
        print(f"  Dropping {y.isnull().sum()} rows with unmapped labels")
        mask = y.notnull()
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

    print(f"  {len(X)} samples, {X.shape[1]} features, target={target_col}")
    return X, y.astype(int), feature_names


def stratified_time_split(X, y, train_ratio=0.70, val_ratio=0.15, min_per_class_test=30):
    """Split chronologically while guaranteeing every class appears in each split.

    For rare classes, forces at least min_per_class_test samples into test and
    a proportional amount into val, ensuring all classes are represented.
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Start with a simple chronological split
    X_train = X.iloc[:train_end].copy()
    X_val = X.iloc[train_end:val_end].copy()
    X_test = X.iloc[val_end:].copy()
    y_train = y.iloc[:train_end].copy()
    y_val = y.iloc[train_end:val_end].copy()
    y_test = y.iloc[val_end:].copy()

    # For each class, ensure minimum representation in test and val
    for cls in range(len(REGIME_CLASSES)):
        test_count = (y_test == cls).sum()
        if test_count >= min_per_class_test:
            continue

        needed = min_per_class_test - test_count
        # Take from the tail of train (closest chronologically)
        cls_in_train = y_train[y_train == cls].index
        if len(cls_in_train) < needed:
            needed = len(cls_in_train)
        if needed == 0:
            continue

        move_idx = cls_in_train[-needed:]
        X_test = pd.concat([X_test, X_train.loc[move_idx]])
        y_test = pd.concat([y_test, y_train.loc[move_idx]])
        X_train = X_train.drop(move_idx)
        y_train = y_train.drop(move_idx)

        # Also move proportional amount to val
        val_needed = max(1, int(needed * val_ratio / (1 - train_ratio)))
        cls_in_train = y_train[y_train == cls].index
        val_needed = min(val_needed, len(cls_in_train))
        if val_needed > 0:
            move_val = cls_in_train[-val_needed:]
            X_val = pd.concat([X_val, X_train.loc[move_val]])
            y_val = pd.concat([y_val, y_train.loc[move_val]])
            X_train = X_train.drop(move_val)
            y_train = y_train.drop(move_val)

    return (
        X_train.reset_index(drop=True), X_val.reset_index(drop=True), X_test.reset_index(drop=True),
        y_train.reset_index(drop=True), y_val.reset_index(drop=True), y_test.reset_index(drop=True),
    )


def stratify_classes(X, y, min_samples=500, max_ratio=10.0):
    """Downsample dominant classes and oversample rare classes to balance the dataset."""
    counts = y.value_counts()
    max_keep = int(counts.max() / max_ratio)
    target_per_class = max(max_keep, min_samples)

    indices = []
    for cls in counts.index:
        cls_idx = list(y[y == cls].index)
        n_have = len(cls_idx)

        if n_have >= target_per_class:
            # Downsample dominant classes
            sampled = np.random.choice(cls_idx, target_per_class, replace=False)
            indices.extend(sampled)
        else:
            # Keep all originals + oversample to reach target
            indices.extend(cls_idx)
            extra = np.random.choice(cls_idx, target_per_class - n_have, replace=True)
            indices.extend(extra)

    np.random.shuffle(indices)
    print(f"  Stratified: {len(y)} -> {len(indices)} samples (target {target_per_class}/class)")
    for cls in sorted(counts.index):
        print(f"    {REGIME_CLASSES[cls]}: {(y.iloc[indices] == cls).sum()}")
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)


def train_ensemble(X_train, y_train, X_val=None, y_val=None):
    """Train XGBoost + LightGBM ensemble, return both models."""
    num_class = len(REGIME_CLASSES)

    print(f"  Training XGBoost on {len(X_train)} samples, {X_train.shape[1]} features")
    start = time.time()

    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="mlogloss",
        early_stopping_rounds=20 if X_val is not None else None,
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
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
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

    print(f"  XGBoost accuracy:  {acc_xgb:.4f}")
    print(f"  LightGBM accuracy: {acc_lgbm:.4f}")
    print(f"  Ensemble accuracy: {acc_ens:.4f}")
    all_labels = list(range(len(REGIME_CLASSES)))
    print(classification_report(y_test, pred_ensemble, labels=all_labels, target_names=REGIME_CLASSES, zero_division=0))

    return acc_ens


def export_models(xgb_model, lgbm_model, feature_names, tf, output_dir):
    """Save models + metadata for one timeframe."""
    out = Path(output_dir) / tf
    out.mkdir(parents=True, exist_ok=True)

    xgb_path = out / "xgb_regime.pkl"
    lgbm_path = out / "lgbm_regime.pkl"
    meta_path = out / "metadata.json"
    features_path = out / "feature_names.json"

    joblib.dump(xgb_model, xgb_path)
    joblib.dump(lgbm_model, lgbm_path)
    json.dump(feature_names, open(features_path, "w"), indent=2)
    json.dump({
        "timeframe": tf,
        "regime_classes": REGIME_CLASSES,
        "model_version": f"v1_{int(time.time())}",
        "xgb_file": xgb_path.name,
        "lgbm_file": lgbm_path.name,
        "feature_file": features_path.name,
        "num_features": len(feature_names),
    }, open(meta_path, "w"), indent=2)

    print(f"\n  Exported to {out.resolve()}/")
    print(f"    {xgb_path.name}  ({xgb_path.stat().st_size / 1024:.0f} KB)")
    print(f"    {lgbm_path.name}  ({lgbm_path.stat().st_size / 1024:.0f} KB)")


def train_timeframe(csv_path: str, tf: str, output_dir: str):
    """Full train + evaluate + export pipeline for one timeframe."""
    print(f"\n{'=' * 60}")
    print(f"  Timeframe: {tf}")
    print(f"{'=' * 60}")

    X, y, feature_names = load_for_timeframe(csv_path, tf)

    X_bal, y_bal = stratify_classes(X, y, min_samples=100, max_ratio=10.0)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_time_split(X_bal, y_bal)
    print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    xgb_model, lgbm_model = train_ensemble(X_train, y_train, X_val, y_val)

    evaluate(xgb_model, lgbm_model, X_test, y_test)

    print(f"\n  Retraining on full balanced data for export...")
    xgb_full, lgbm_full = train_ensemble(X_bal, y_bal)
    export_models(xgb_full, lgbm_full, feature_names, tf, output_dir)


def load_combined(csv_path: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load CSV with all 15m + 1h features combined, using 15m_regime as target."""
    df = pd.read_csv(csv_path)

    exclude = ALL_LABEL_COLUMNS | {"time"}
    feature_cols = [c for c in df.columns if c not in exclude]
    feature_names = feature_cols

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    y = df["15m_regime"].map({name: i for i, name in enumerate(REGIME_CLASSES)})

    if y.isnull().any():
        print(f"  Dropping {y.isnull().sum()} rows with unmapped labels")
        mask = y.notnull()
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

    print(f"  {len(X)} samples, {X.shape[1]} features (15m + 1h combined)")
    return X, y.astype(int), feature_names


def train_combined(csv_path: str, output_dir: str):
    """Train a single model on all features from both timeframes."""
    print(f"\n{'=' * 60}")
    print(f"  Combined (15m + 1h features)")
    print(f"{'=' * 60}")

    X, y, feature_names = load_combined(csv_path)

    X_bal, y_bal = stratify_classes(X, y, min_samples=100, max_ratio=10.0)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_time_split(X_bal, y_bal)
    print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    xgb_model, lgbm_model = train_ensemble(X_train, y_train, X_val, y_val)

    evaluate(xgb_model, lgbm_model, X_test, y_test)

    print(f"\n  Retraining on full balanced data for export...")
    xgb_full, lgbm_full = train_ensemble(X_bal, y_bal)
    export_models(xgb_full, lgbm_full, feature_names, "combined", output_dir)


def main(csv_path="hmm_regime_labels.csv", output_dir="trained_output"):
    print("=" * 60)
    print("Regime Classification — Multi-Timeframe Training")
    print("=" * 60)

    for tf in TIMEFRAMES:
        train_timeframe(csv_path, tf, output_dir)

    train_combined(csv_path, output_dir)

    print(f"\n{'=' * 60}")
    print(f"All done. Output: {Path(output_dir).resolve()}/")
    print(f"  15m/      — per-timeframe 15m model")
    print(f"  1h/       — per-timeframe 1h model")
    print(f"  combined/ — single model with all 15m + 1h features")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="hmm_regime_labels.csv")
    parser.add_argument("--output", default="trained_output")
    args = parser.parse_args()
    main(args.csv, args.output)
