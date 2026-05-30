# ============================================================
# STEP 6 — HYPERPARAMETER TUNING
# Project : Watch Price Prediction (WatchVine Dataset)
# Input   : watchvine_train_X.csv, watchvine_train_y.csv,
#           watchvine_test_X.csv, watchvine_test_y.csv
# Output  : final_model.pkl, tuning plots
# Author  : Rudra (GitHub: Rudra2986)
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("tuning_plots", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("STEP 6 — HYPERPARAMETER TUNING")
print("=" * 60)

X_train = pd.read_csv("watchvine_train_X.csv")
y_train = pd.read_csv("watchvine_train_y.csv")["log_price"]
X_test  = pd.read_csv("watchvine_test_X.csv")
y_test  = pd.read_csv("watchvine_test_y.csv")
y_test_log  = y_test["log_price"]
y_test_orig = y_test["price"]

print(f"\n  Train: {X_train.shape} | Test: {X_test.shape}")

# Load top model names from Step 5
top_models = joblib.load("top_models.pkl")
print(f"  Top models from Step 5: {top_models}")


# ============================================================
# DEFINE OPTUNA OBJECTIVES
# ============================================================

def xgboost_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 10.0),
        "random_state":     42,
        "verbosity":        0,
    }
    model = XGBRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
    return score


def lightgbm_objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":         trial.suggest_int("max_depth", 3, 15),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "random_state":      42,
        "verbose":           -1,
    }
    model = LGBMRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
    return score


def catboost_objective(trial):
    params = {
        "iterations":     trial.suggest_int("iterations", 100, 1000),
        "depth":          trial.suggest_int("depth", 3, 10),
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg":    trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "random_state":   42,
        "verbose":        0,
    }
    model = CatBoostRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
    return score


# Map model names to objectives and constructors
objective_map = {
    "XGBoost":  (xgboost_objective,  XGBRegressor),
    "LightGBM": (lightgbm_objective, LGBMRegressor),
    "CatBoost": (catboost_objective, CatBoostRegressor),
}


# ============================================================
# RUN OPTUNA TUNING
# ============================================================

N_TRIALS = 100
best_models = {}

for model_name in top_models:
    if model_name not in objective_map:
        print(f"\n  Skipping {model_name} — no Optuna objective defined (linear model)")
        print(f"  Using baseline model from Step 4 instead.")
        best_models[model_name] = joblib.load(f"model_{model_name}.pkl")
        continue

    objective_fn, ModelClass = objective_map[model_name]

    print(f"\n" + "-" * 60)
    print(f"TUNING: {model_name} ({N_TRIALS} trials)")
    print("-" * 60)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"  Best R2 (CV): {study.best_value:.4f}")
    print(f"  Best params : {study.best_params}")

    # Retrain with best params on full training set
    best_params = study.best_params.copy()
    best_params["random_state"] = 42

    if model_name == "XGBoost":
        best_params["verbosity"] = 0
        final_model = XGBRegressor(**best_params)
    elif model_name == "LightGBM":
        best_params["verbose"] = -1
        final_model = LGBMRegressor(**best_params)
    elif model_name == "CatBoost":
        best_params["verbose"] = 0
        final_model = CatBoostRegressor(**best_params)

    final_model.fit(X_train, y_train)
    best_models[model_name] = final_model

    # Save Optuna study visualization
    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title(f'{model_name} — Optimization History', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'tuning_plots/6_1_{model_name}_optimization_history.png', dpi=150)
        plt.close()
        print(f"  Saved: tuning_plots/6_1_{model_name}_optimization_history.png")
    except Exception:
        pass

    try:
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title(f'{model_name} — Parameter Importance', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'tuning_plots/6_2_{model_name}_param_importance.png', dpi=150)
        plt.close()
        print(f"  Saved: tuning_plots/6_2_{model_name}_param_importance.png")
    except Exception:
        pass


# ============================================================
# EVALUATE TUNED MODELS ON TEST SET
# ============================================================

print("\n" + "=" * 60)
print("TUNED MODEL EVALUATION ON TEST SET")
print("=" * 60)

best_r2 = -np.inf
best_name = None
best_final_model = None

for name, model in best_models.items():
    preds_log  = model.predict(X_test)
    preds_orig = np.expm1(preds_log)

    r2   = r2_score(y_test_log, preds_log)
    rmse = np.sqrt(mean_squared_error(y_test_log, preds_log))
    mae  = mean_absolute_error(y_test_log, preds_log)
    rmse_rs = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
    mae_rs  = mean_absolute_error(y_test_orig, preds_orig)

    print(f"\n  {name} (tuned):")
    print(f"    R2         = {r2:.4f}")
    print(f"    RMSE (log) = {rmse:.4f}")
    print(f"    MAE  (log) = {mae:.4f}")
    print(f"    RMSE (Rs.) = Rs.{rmse_rs:.2f}")
    print(f"    MAE  (Rs.) = Rs.{mae_rs:.2f}")

    if r2 > best_r2:
        best_r2 = r2
        best_name = name
        best_final_model = model


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print(f"\n  FINAL MODEL: {best_name} (R2 = {best_r2:.4f})")

joblib.dump(best_final_model, "final_model.pkl")
print(f"  Saved: final_model.pkl")

# Save model metadata for the Streamlit app
model_metadata = {
    "model_name": best_name,
    "r2": best_r2,
    "rmse_log": np.sqrt(mean_squared_error(y_test_log, best_final_model.predict(X_test))),
    "rmse_rs": np.sqrt(mean_squared_error(y_test_orig, np.expm1(best_final_model.predict(X_test)))),
}
joblib.dump(model_metadata, "model_metadata.pkl")
print(f"  Saved: model_metadata.pkl")


# ============================================================
# SHAP FEATURE IMPORTANCE
# ============================================================

print("\n" + "-" * 60)
print("SHAP FEATURE IMPORTANCE")
print("-" * 60)

try:
    import shap

    # Use a sample for SHAP (faster)
    X_sample = X_train.sample(n=min(200, len(X_train)), random_state=42)

    explainer = shap.TreeExplainer(best_final_model)
    shap_values = explainer.shap_values(X_sample)

    # SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title(f'{best_name} — SHAP Feature Importance (Top 20)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tuning_plots/6_3_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: tuning_plots/6_3_shap_summary.png")

    # SHAP Bar Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title(f'{best_name} — Mean |SHAP| (Top 20 Features)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tuning_plots/6_4_shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: tuning_plots/6_4_shap_bar.png")

except Exception as e:
    print(f"  SHAP analysis skipped: {e}")
    print("  (This is non-critical — model is still saved.)")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 6 COMPLETE")
print("=" * 60)
print(f"  Models tuned     : {list(best_models.keys())}")
print(f"  Optuna trials    : {N_TRIALS} per model")
print(f"  Final model      : {best_name}")
print(f"  Final R2 (test)  : {best_r2:.4f}")
print(f"  Saved            : final_model.pkl, model_metadata.pkl")
print(f"  SHAP plots in    : tuning_plots/")
print(f"\n  Next: Step 7 — Streamlit Deployment")
print("=" * 60)
