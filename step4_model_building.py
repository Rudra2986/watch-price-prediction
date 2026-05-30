# ============================================================
# STEP 4 — MODEL BUILDING
# Project : Watch Price Prediction (WatchVine Dataset)
# Input   : watchvine_train_X.csv, watchvine_train_y.csv
# Output  : model_*.pkl files (one per model)
# Author  : Rudra (GitHub: Rudra2986)
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("STEP 4 — MODEL BUILDING")
print("=" * 60)

X_train = pd.read_csv("watchvine_train_X.csv")
y_train = pd.read_csv("watchvine_train_y.csv")["log_price"]

print(f"\n  Loaded: X_train = {X_train.shape}")
print(f"  Loaded: y_train = {y_train.shape}")
print(f"  Target: log_price (mean={y_train.mean():.4f}, std={y_train.std():.4f})")


# ============================================================
# DEFINE MODELS
# ============================================================
# Linear Regression & Ridge use StandardScaler via Pipeline
# because they are NOT scale-invariant.
# Tree-based models don't need scaling.
# ============================================================

print("\n" + "-" * 60)
print("DEFINING MODELS")
print("-" * 60)

models = {
    "Linear_Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression())
    ]),

    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0))
    ]),

    "Random_Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        random_state=42,
        verbosity=0
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=100,
        random_state=42,
        verbose=-1
    ),

    "CatBoost": CatBoostRegressor(
        iterations=100,
        random_state=42,
        verbose=0
    ),
}

print(f"  Models defined: {list(models.keys())}")


# ============================================================
# TRAIN & EVALUATE WITH 5-FOLD CROSS-VALIDATION
# ============================================================
# We use 5-fold CV on the TRAIN set only.
# Test set is NOT touched — that's for Step 5.
# ============================================================

print("\n" + "-" * 60)
print("5-FOLD CROSS-VALIDATION (on train set only)")
print("-" * 60)

results = {}

for name, model in models.items():
    print(f"\n  Training: {name}...")

    # 5-fold CV — R2 scoring
    cv_r2 = cross_val_score(model, X_train, y_train,
                            cv=5, scoring="r2")

    # 5-fold CV — negative RMSE (sklearn convention)
    cv_rmse = cross_val_score(model, X_train, y_train,
                              cv=5, scoring="neg_root_mean_squared_error")

    results[name] = {
        "R2_mean": cv_r2.mean(),
        "R2_std":  cv_r2.std(),
        "RMSE_mean": -cv_rmse.mean(),   # negate back to positive
        "RMSE_std":  cv_rmse.std(),
    }

    print(f"    R2   = {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print(f"    RMSE = {-cv_rmse.mean():.4f} +/- {cv_rmse.std():.4f}")

    # Fit on full training set for saving
    model.fit(X_train, y_train)

    # Save model
    filename = f"model_{name}.pkl"
    joblib.dump(model, filename)
    print(f"    Saved: {filename}")


# ============================================================
# RESULTS SUMMARY TABLE
# ============================================================

print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS SUMMARY")
print("=" * 60)

# Header
print(f"\n  {'Model':<22} {'R2 (mean)':>10} {'R2 (std)':>10} {'RMSE (mean)':>12} {'RMSE (std)':>11}")
print("  " + "-" * 67)

# Sort by R2 descending
sorted_results = sorted(results.items(), key=lambda x: x[1]["R2_mean"], reverse=True)

for name, metrics in sorted_results:
    print(f"  {name:<22} {metrics['R2_mean']:>10.4f} {metrics['R2_std']:>10.4f} "
          f"{metrics['RMSE_mean']:>12.4f} {metrics['RMSE_std']:>11.4f}")


# ============================================================
# IDENTIFY BEST MODELS
# ============================================================

best_name = sorted_results[0][0]
second_name = sorted_results[1][0]

print(f"\n  Best model   : {best_name} (R2 = {sorted_results[0][1]['R2_mean']:.4f})")
print(f"  Runner-up    : {second_name} (R2 = {sorted_results[1][1]['R2_mean']:.4f})")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 4 COMPLETE")
print("=" * 60)
print(f"  Models trained: {len(models)}")
print(f"  CV folds      : 5")
print(f"  Target        : log_price")
print(f"  Files saved   : {[f'model_{n}.pkl' for n in models.keys()]}")
print(f"\n  Next: Step 5 — Model Evaluation (test set)")
print("=" * 60)
