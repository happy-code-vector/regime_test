"""Shared training utilities for all train scripts.

Provides:
  - Optuna hyperparameter optimization
  - train_ensemble with static or tuned params
  - evaluate with classification report + confusion matrix
  - export_models with metadata
  - chronological_split, stratify_classes, smooth_labels
  - compute_sample_weights, add_engineered_features
"""

import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgbm
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib


# ──────────────────────────────────────────────────────────────
# Optuna
# ──────────────────────────────────────────────────────────────

def run_optuna_study(X_train, y_train, X_val, y_val, num_class,
                     sample_weight=None, n_trials=100):
    """Run Optuna to find best XGBoost + LightGBM hyperparameters.

    Uses separate studies per model (8D + 9D instead of 17D joint),
    multivariate TPE sampler, and early stopping on both models.
    Returns (best_xgb_params, best_lgbm_params) dicts.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)

    # ── XGBoost study (8 dimensions) ──
    print("\n  [Optuna] Tuning XGBoost...")
    def xgb_objective(trial):
        params = {
            "objective": "multi:softprob",
            "num_class": num_class,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child", 1, 10),
            "reg_alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "eval_metric": "mlogloss",
            "early_stopping_rounds": 25,
            "verbosity": 0,
        }
        model = xgb.XGBClassifier(**params)
        fit_kw = {"eval_set": [(X_val.values, y_val.values)], "verbose": False}
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight
        model.fit(X_train.values, y_train.values, **fit_kw)
        return f1_score(y_val.values, model.predict(X_val.values), average="macro")

    xgb_study = optuna.create_study(direction="maximize", sampler=sampler)
    xgb_study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=True)

    xgb_best = {
        "n_estimators": xgb_study.best_trial.params["n_estimators"],
        "max_depth": xgb_study.best_trial.params["max_depth"],
        "learning_rate": xgb_study.best_trial.params["lr"],
        "subsample": xgb_study.best_trial.params["subsample"],
        "colsample_bytree": xgb_study.best_trial.params["colsample"],
        "min_child_weight": xgb_study.best_trial.params["min_child"],
        "reg_alpha": xgb_study.best_trial.params["alpha"],
        "reg_lambda": xgb_study.best_trial.params["lambda"],
    }
    print(f"  [Optuna] Best XGBoost val macro-F1: {xgb_study.best_value:.4f}")

    # ── LightGBM study (9 dimensions, num_leaves constrained by max_depth) ──
    print("\n  [Optuna] Tuning LightGBM...")
    def lgbm_objective(trial):
        max_depth = trial.suggest_int("max_depth", 3, 12)
        max_leaves = min(300, 2 ** max_depth)  # num_leaves can't exceed 2^max_depth
        params = {
            "objective": "multiclass",
            "num_class": num_class,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": max_depth,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 20, max_leaves),
            "min_child_samples": trial.suggest_int("min_child", 5, 100),
            "reg_alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "verbose": -1,
        }
        model = lgbm.LGBMClassifier(**params)
        fit_kw = {"eval_set": [(X_val.values, y_val.values)],
                  "callbacks": [lgbm.early_stopping(25, verbose=False)]}
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight
        model.fit(X_train.values, y_train.values, **fit_kw)
        return f1_score(y_val.values, model.predict(X_val.values), average="macro")

    lgbm_study = optuna.create_study(direction="maximize", sampler=sampler)
    lgbm_study.optimize(lgbm_objective, n_trials=n_trials, show_progress_bar=True)

    lgbm_best = {
        "n_estimators": lgbm_study.best_trial.params["n_estimators"],
        "max_depth": lgbm_study.best_trial.params["max_depth"],
        "learning_rate": lgbm_study.best_trial.params["lr"],
        "subsample": lgbm_study.best_trial.params["subsample"],
        "colsample_bytree": lgbm_study.best_trial.params["colsample"],
        "num_leaves": lgbm_study.best_trial.params["num_leaves"],
        "min_child_samples": lgbm_study.best_trial.params["min_child"],
        "reg_alpha": lgbm_study.best_trial.params["alpha"],
        "reg_lambda": lgbm_study.best_trial.params["lambda"],
    }
    print(f"  [Optuna] Best LightGBM val macro-F1: {lgbm_study.best_value:.4f}")

    # ── Final ensemble eval with best params ──
    print("\n  [Optuna] Evaluating best ensemble...")
    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=num_class,
        eval_metric="mlogloss", early_stopping_rounds=25, verbosity=0,
        **xgb_best,
    )
    fit_kw = {"eval_set": [(X_val.values, y_val.values)], "verbose": False}
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight
    xgb_model.fit(X_train.values, y_train.values, **fit_kw)

    lgbm_model = lgbm.LGBMClassifier(
        objective="multiclass", num_class=num_class, verbose=-1,
        **lgbm_best,
    )
    lgbm_kw = {"eval_set": [(X_val.values, y_val.values)],
               "callbacks": [lgbm.early_stopping(25, verbose=False)]}
    if sample_weight is not None:
        lgbm_kw["sample_weight"] = sample_weight
    lgbm_model.fit(X_train.values, y_train.values, **lgbm_kw)

    proba_xgb = xgb_model.predict_proba(X_val.values)
    proba_lgbm = lgbm_model.predict_proba(X_val.values)
    pred = ((proba_xgb + proba_lgbm) / 2).argmax(axis=1)
    ensemble_f1 = f1_score(y_val.values, pred, average="macro")

    print(f"\n  Best ensemble val macro-F1: {ensemble_f1:.4f}")
    print(f"  XGBoost params: {xgb_best}")
    print(f"  LightGBM params: {lgbm_best}")

    return xgb_best, lgbm_best


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

STATIC_XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

STATIC_LGBM_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def train_ensemble(X_train, y_train, X_val=None, y_val=None,
                   num_class=5, xgb_params=None, lgbm_params=None,
                   sample_weight=None):
    """Train XGBoost + LightGBM ensemble.

    If xgb_params/lgbm_params are None, uses static defaults.
    Pass sample_weight for weighted training.
    """
    xgb_p = xgb_params or STATIC_XGB_PARAMS
    lgbm_p = lgbm_params or STATIC_LGBM_PARAMS

    print(f"\n  Training XGBoost on {len(X_train)} samples, {X_train.shape[1]} features")
    start = time.time()

    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        early_stopping_rounds=25 if X_val is not None else None,
        verbosity=0,
        **xgb_p,
    )
    fit_kw = {}
    if X_val is not None:
        fit_kw["eval_set"] = [(X_val.values, y_val.values)]
        fit_kw["verbose"] = False
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight
    xgb_model.fit(X_train.values, y_train.values, **fit_kw)
    print(f"  XGBoost trained in {time.time() - start:.1f}s")

    print(f"  Training LightGBM on {len(X_train)} samples")
    start = time.time()

    lgbm_model = lgbm.LGBMClassifier(
        objective="multiclass",
        num_class=num_class,
        verbose=-1,
        **lgbm_p,
    )
    lgbm_kw = {}
    if X_val is not None:
        lgbm_kw["eval_set"] = [(X_val.values, y_val.values)]
        lgbm_kw["callbacks"] = [lgbm.early_stopping(25, verbose=False)]
    if sample_weight is not None:
        lgbm_kw["sample_weight"] = sample_weight
    lgbm_model.fit(X_train.values, y_train.values, **lgbm_kw)
    print(f"  LightGBM trained in {time.time() - start:.1f}s")

    return xgb_model, lgbm_model


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(xgb_model, lgbm_model, X_test, y_test, regime_classes, tag=""):
    """Evaluate ensemble — prints accuracy, macro-F1, classification report, confusion matrix."""
    proba_xgb = xgb_model.predict_proba(X_test.values)
    proba_lgbm = lgbm_model.predict_proba(X_test.values)
    proba_ensemble = (proba_xgb + proba_lgbm) / 2
    pred_ensemble = proba_ensemble.argmax(axis=1)

    acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test.values))
    acc_lgbm = accuracy_score(y_test, lgbm_model.predict(X_test.values))
    acc_ens = accuracy_score(y_test, pred_ensemble)
    f1_ens = f1_score(y_test, pred_ensemble, average="macro")

    print(f"\n  {tag}XGBoost accuracy:  {acc_xgb:.4f}")
    print(f"  {tag}LightGBM accuracy: {acc_lgbm:.4f}")
    print(f"  {tag}Ensemble accuracy: {acc_ens:.4f}")
    print(f"  {tag}Ensemble macro-F1: {f1_ens:.4f}")

    present_labels = sorted(y_test.unique())
    present_names = [regime_classes[i] for i in present_labels]
    print(classification_report(
        y_test, pred_ensemble,
        labels=present_labels, target_names=present_names,
        zero_division=0,
    ))

    cm = confusion_matrix(y_test, pred_ensemble, labels=range(len(regime_classes)))
    print(f"  {tag}Confusion matrix (rows=true, cols=pred):")
    col_w = max(len(n) for n in regime_classes) + 2
    header = " " * (col_w + 2) + "  ".join([f"{n:>{col_w}s}" for n in regime_classes])
    print(header)
    for i in range(len(regime_classes)):
        row = "  ".join([f"{cm[i,j]:>{col_w}d}" for j in range(len(regime_classes))])
        print(f"  {regime_classes[i]:{col_w}s}: {row}")

    return acc_ens, f1_ens


# ──────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────

def export_models(xgb_model, lgbm_model, feature_names, output_dir,
                  timeframe, regime_classes, metadata_extra=None):
    """Save models + metadata + feature names."""
    out = Path(output_dir) / timeframe
    out.mkdir(parents=True, exist_ok=True)

    xgb_path = out / "xgb_regime.pkl"
    lgbm_path = out / "lgbm_regime.pkl"
    meta_path = out / "metadata.json"
    features_path = out / "feature_names.json"

    joblib.dump(xgb_model, xgb_path)
    joblib.dump(lgbm_model, lgbm_path)
    json.dump(feature_names, open(features_path, "w"), indent=2)

    meta = {
        "timeframe": timeframe,
        "regime_classes": regime_classes,
        "model_version": f"v1_{int(time.time())}",
        "xgb_file": xgb_path.name,
        "lgbm_file": lgbm_path.name,
        "feature_file": features_path.name,
        "num_features": len(feature_names),
    }
    if metadata_extra:
        meta.update(metadata_extra)
    json.dump(meta, open(meta_path, "w"), indent=2)

    print(f"\n  Exported to {out.resolve()}\\")
    print(f"    {xgb_path.name}  ({xgb_path.stat().st_size / 1024:.0f} KB)")
    print(f"    {lgbm_path.name}  ({lgbm_path.stat().st_size / 1024:.0f} KB)")


# ──────────────────────────────────────────────────────────────
# Data utilities
# ──────────────────────────────────────────────────────────────

def chronological_split(X, y, train_ratio=0.70, val_ratio=0.15):
    """Simple chronological split — data must be sorted by time first."""
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

    return splits


def stratify_classes(X, y, min_samples=1000, max_ratio=5.0, regime_classes=None):
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
        name = regime_classes[cls] if regime_classes else str(cls)
        print(f"    {name:12s}: {(y.iloc[indices] == cls).sum()}")
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)


def smooth_labels(y, window=7):
    """Rolling majority vote to stabilize noisy labels."""
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


def compute_sample_weights(y):
    """Compute per-sample weights from class frequencies (balanced)."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_map = dict(zip(classes, weights))
    sample_weights = np.array([class_weight_map[yi] for yi in y])
    print(f"  Class weights:")
    for cls, w in sorted(class_weight_map.items()):
        print(f"    class {cls}: {w:.3f}")
    return sample_weights


def add_engineered_features(X, core_features):
    """Add interaction, lag, and rolling features."""
    df = X.copy()

    # Cross-feature interactions
    for col_a, col_b in [
        ("slope_z_5m", "adx_z_5m"),
        ("slope_z_15m", "adx_z_15m"),
        ("slope_z_5m", "realized_vol_z_5m"),
        ("adx_z_5m", "realized_vol_z_5m"),
    ]:
        if col_a in df.columns and col_b in df.columns:
            df[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]

    # Slope/adx diffs across timeframes
    for a, b in [("slope_z_5m", "slope_z_15m"), ("adx_z_5m", "adx_z_15m")]:
        if a in df.columns and b in df.columns:
            df[f"{a}_minus_{b}"] = df[a] - df[b]

    # Lag features (previous 1, 2, 3 bars)
    for col in core_features:
        if col in df.columns:
            for lag in [1, 2, 3]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling stats (5-bar window)
    for col in core_features:
        if col in df.columns:
            df[f"{col}_rmean5"] = df[col].rolling(5, min_periods=1).mean()
            df[f"{col}_rstd5"] = df[col].rolling(5, min_periods=1).std().fillna(0)

    df = df.fillna(0).astype(np.float32)
    print(f"  Engineered features: {X.shape[1]} -> {df.shape[1]} (+{df.shape[1] - X.shape[1]})")
    return df
