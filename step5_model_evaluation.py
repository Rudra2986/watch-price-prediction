# ============================================================
# STEP 5 — MODEL EVALUATION
# Project : Watch Price Prediction (WatchVine Dataset)
# Input   : watchvine_test_X.csv, watchvine_test_y.csv, model_*.pkl
# Output  : Evaluation metrics, comparison table, plots
# Author  : Rudra (GitHub: Rudra2986)
# ============================================================

import pandas as pd
import numpy as np
import joblib
import glob
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("evaluation_plots", exist_ok=True)


# ============================================================
# LOAD DATA & MODELS
# ============================================================

print("=" * 60)
print("STEP 5 — MODEL EVALUATION")
print("=" * 60)

X_test = pd.read_csv("watchvine_test_X.csv")
y_test = pd.read_csv("watchvine_test_y.csv")

y_log  = y_test["log_price"]
y_orig = y_test["price"]

print(f"\n  Test set: {X_test.shape[0]} rows x {X_test.shape[1]} features")
print(f"  Price range (test): Rs.{y_orig.min():.0f} - Rs.{y_orig.max():.0f}")

# Load all saved models
model_files = sorted(glob.glob("model_*.pkl"))
# Exclude metadata file
model_files = [f for f in model_files if "metadata" not in f]
print(f"\n  Found {len(model_files)} model files: {model_files}")

trained_models = {}
for f in model_files:
    name = f.replace("model_", "").replace(".pkl", "")
    trained_models[name] = joblib.load(f)
    print(f"    Loaded: {name}")


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

print("\n" + "-" * 60)
print("EVALUATION ON TEST SET")
print("-" * 60)

results = {}

for name, model in trained_models.items():
    # Predict
    preds_log  = model.predict(X_test)
    preds_orig = np.expm1(preds_log)

    # Metrics on log scale
    r2   = r2_score(y_log, preds_log)
    rmse = np.sqrt(mean_squared_error(y_log, preds_log))
    mae  = mean_absolute_error(y_log, preds_log)

    # Metrics on original Rs. scale
    rmse_rs = np.sqrt(mean_squared_error(y_orig, preds_orig))
    mae_rs  = mean_absolute_error(y_orig, preds_orig)

    results[name] = {
        "R2":        r2,
        "RMSE_log":  rmse,
        "MAE_log":   mae,
        "RMSE_Rs":   rmse_rs,
        "MAE_Rs":    mae_rs,
        "preds_log": preds_log,
        "preds_orig": preds_orig,
    }

    print(f"\n  {name}:")
    print(f"    R2         = {r2:.4f}")
    print(f"    RMSE (log) = {rmse:.4f}")
    print(f"    MAE  (log) = {mae:.4f}")
    print(f"    RMSE (Rs.) = Rs.{rmse_rs:.2f}")
    print(f"    MAE  (Rs.) = Rs.{mae_rs:.2f}")


# ============================================================
# COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON TABLE")
print("=" * 60)

print(f"\n  {'Model':<22} {'R2':>8} {'RMSE(log)':>10} {'MAE(log)':>9} {'RMSE(Rs.)':>10} {'MAE(Rs.)':>9}")
print("  " + "-" * 70)

sorted_results = sorted(results.items(), key=lambda x: x[1]["R2"], reverse=True)

for name, m in sorted_results:
    print(f"  {name:<22} {m['R2']:>8.4f} {m['RMSE_log']:>10.4f} {m['MAE_log']:>9.4f} "
          f"{m['RMSE_Rs']:>10.2f} {m['MAE_Rs']:>9.2f}")


# ============================================================
# TOP 2 MODELS
# ============================================================

top1_name = sorted_results[0][0]
top2_name = sorted_results[1][0]

print(f"\n  Top 1: {top1_name} (R2 = {sorted_results[0][1]['R2']:.4f})")
print(f"  Top 2: {top2_name} (R2 = {sorted_results[1][1]['R2']:.4f})")
print(f"\n  These 2 models will go to Step 6 for hyperparameter tuning.")

# Save top model names for Step 6 reference
joblib.dump([top1_name, top2_name], "top_models.pkl")
print(f"  Saved: top_models.pkl")


# ============================================================
# PLOT 1: R2 COMPARISON BAR CHART
# ============================================================

print("\n" + "-" * 60)
print("GENERATING EVALUATION PLOTS")
print("-" * 60)

model_names = [name for name, _ in sorted_results]
r2_scores   = [m["R2"] for _, m in sorted_results]

plt.figure(figsize=(12, 6))
colors = ['gold' if i == 0 else 'steelblue' if i == 1 else 'lightblue'
          for i in range(len(model_names))]
bars = plt.barh(model_names[::-1], r2_scores[::-1], color=colors[::-1])

for bar, score in zip(bars, r2_scores[::-1]):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f'{score:.4f}', ha='left', va='center', fontweight='bold')

plt.xlabel('R2 Score')
plt.title('Model Comparison — R2 Score on Test Set', fontsize=14, fontweight='bold')
plt.xlim(0, max(r2_scores) * 1.15)
plt.tight_layout()
plt.savefig('evaluation_plots/5_1_r2_comparison.png', dpi=150)
plt.close()
print("  Saved: evaluation_plots/5_1_r2_comparison.png")


# ============================================================
# PLOT 2: RMSE COMPARISON (Rs. scale)
# ============================================================

rmse_scores = [m["RMSE_Rs"] for _, m in sorted_results]

plt.figure(figsize=(12, 6))
colors = ['gold' if i == 0 else 'steelblue' if i == 1 else 'lightblue'
          for i in range(len(model_names))]
bars = plt.barh(model_names[::-1], rmse_scores[::-1], color=colors[::-1])

for bar, score in zip(bars, rmse_scores[::-1]):
    plt.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
             f'Rs.{score:.0f}', ha='left', va='center', fontweight='bold')

plt.xlabel('RMSE (Rs.)')
plt.title('Model Comparison — RMSE in Rs. (Test Set)', fontsize=14, fontweight='bold')
plt.xlim(0, max(rmse_scores) * 1.2)
plt.tight_layout()
plt.savefig('evaluation_plots/5_2_rmse_comparison.png', dpi=150)
plt.close()
print("  Saved: evaluation_plots/5_2_rmse_comparison.png")


# ============================================================
# PLOT 3: ACTUAL vs PREDICTED (top 2 models)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, name in zip(axes, [top1_name, top2_name]):
    preds = results[name]["preds_orig"]
    ax.scatter(y_orig, preds, alpha=0.5, s=20, color='steelblue')
    ax.plot([y_orig.min(), y_orig.max()],
            [y_orig.min(), y_orig.max()],
            'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual Price (Rs.)')
    ax.set_ylabel('Predicted Price (Rs.)')
    ax.set_title(f'{name}\nR2={results[name]["R2"]:.4f} | RMSE=Rs.{results[name]["RMSE_Rs"]:.0f}')
    ax.legend()

plt.suptitle('Actual vs Predicted — Top 2 Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('evaluation_plots/5_3_actual_vs_predicted.png', dpi=150)
plt.close()
print("  Saved: evaluation_plots/5_3_actual_vs_predicted.png")


# ============================================================
# PLOT 4: RESIDUAL PLOTS (top 2 models)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, name in zip(axes, [top1_name, top2_name]):
    preds = results[name]["preds_orig"]
    residuals = y_orig.values - preds
    ax.scatter(preds, residuals, alpha=0.5, s=20, color='tomato')
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Predicted Price (Rs.)')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title(f'{name} — Residuals')

plt.suptitle('Residual Analysis — Top 2 Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('evaluation_plots/5_4_residuals.png', dpi=150)
plt.close()
print("  Saved: evaluation_plots/5_4_residuals.png")


# ============================================================
# PLOT 5: ERROR DISTRIBUTION (top 2 models)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, name in zip(axes, [top1_name, top2_name]):
    preds = results[name]["preds_orig"]
    errors = y_orig.values - preds
    sns.histplot(errors, bins=30, kde=True, color='steelblue', ax=ax)
    ax.axvline(x=0, color='red', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Prediction Error (Rs.)')
    ax.set_ylabel('Count')
    ax.set_title(f'{name} — Error Distribution')

plt.suptitle('Prediction Error Distribution — Top 2 Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('evaluation_plots/5_5_error_distribution.png', dpi=150)
plt.close()
print("  Saved: evaluation_plots/5_5_error_distribution.png")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 5 COMPLETE")
print("=" * 60)
print(f"  Models evaluated : {len(trained_models)}")
print(f"  Test set size    : {len(X_test)} rows")
print(f"  Top 1            : {top1_name} (R2={sorted_results[0][1]['R2']:.4f})")
print(f"  Top 2            : {top2_name} (R2={sorted_results[1][1]['R2']:.4f})")
print(f"  Plots saved in   : evaluation_plots/")
print(f"\n  Next: Step 6 — Hyperparameter Tuning")
print("=" * 60)
