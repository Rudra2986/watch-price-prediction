# ============================================================
# STEP 3 — FEATURE ENGINEERING
# Project : Watch Price Prediction (WatchVine Dataset)
# Input   : watchvine_cleaned.csv
# Output  : watchvine_features.csv
# Author  : Rudra (GitHub: Rudra2986)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("STEP 3 — FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv("watchvine_cleaned.csv")
print(f"\n✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")


# ============================================================
# TASK 1 — DROP REDUNDANT COLUMNS
# Reason:
#   - ai_is_automatic : 100% identical to is_automatic (EDA finding)
#   - ai_watch_type   : duplicate of watch_type
#   - name            : free text, no ML signal
# ============================================================

print("\n" + "─" * 60)
print("TASK 1 — Dropping Redundant Columns")
print("─" * 60)

cols_to_drop = ["ai_is_automatic", "ai_watch_type", "name"]

# Safety check — only drop if they exist
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

print(f"✅ Dropped: {cols_to_drop}")
print(f"   Shape after drop: {df.shape}")


# ============================================================
# TASK 2 — CREATE FLAG COLUMNS
# Why: High unknown% means we can't just ignore unknowns.
#      Instead, we flag whether each row HAS real data.
#      This gives the model useful signal about data quality.
# ============================================================

print("\n" + "─" * 60)
print("TASK 2 — Creating Flag Columns")
print("─" * 60)

# Flag: does this watch have a known brand?
df["has_brand"] = (df["brand"].str.lower() != "unknown").astype(int)

# Flag: does this watch have attribute details (color etc.)?
df["has_details"] = (df["color"].str.lower() != "unknown").astype(int)

# Flag: do we know the watch type?
df["has_watch_type"] = (df["watch_type"].str.lower() != "unknown").astype(int)

print(f"✅ Created: has_brand     → {df['has_brand'].sum()} watches have a known brand")
print(f"✅ Created: has_details   → {df['has_details'].sum()} watches have detail info")
print(f"✅ Created: has_watch_type → {df['has_watch_type'].sum()} watches have known type")


# ============================================================
# TASK 3 — FIX is_automatic COLUMN
# Current values: 'yes' / 'no' / 'unknown'
# Convert to:     1    /  0  /   -1
# Why -1 for unknown? It keeps "no info" distinct from
# "definitely not automatic" (0). Tree models handle -1 fine.
# ============================================================

print("\n" + "─" * 60)
print("TASK 3 — Fixing is_automatic Column")
print("─" * 60)

print(f"   Before: {df['is_automatic'].value_counts().to_dict()}")

auto_map = {"yes": 1, "no": 0, "unknown": -1}
df["is_automatic"] = df["is_automatic"].str.lower().map(auto_map)

# Fallback: anything not mapped → -1
df["is_automatic"] = df["is_automatic"].fillna(-1).astype(int)

print(f"   After : {df['is_automatic'].value_counts().to_dict()}")
print("✅ is_automatic encoded: yes→1, no→0, unknown→-1")


# ============================================================
# TASK 4 — HANDLE HIGH UNKNOWN% COLUMNS
# Decision: Treat 'unknown' as its own valid category.
# Do NOT impute — replacing 'unknown' with the mode would
# inject false information into 57–89% of rows.
# 'unknown' itself carries signal: "no data available".
# ============================================================

print("\n" + "─" * 60)
print("TASK 4 — Confirming 'unknown' Treatment")
print("─" * 60)

categorical_cols = [
    "category", "brand", "color", "style", "material",
    "gender", "dial_color", "strap_material", "strap_color",
    "case_material", "belt_type", "watch_type",
    "watch_style_category"
]

for col in categorical_cols:
    if col in df.columns:
        unknown_count = (df[col].str.lower() == "unknown").sum()
        pct = unknown_count / len(df) * 100
        print(f"   {col:<25} unknown: {unknown_count:>4} ({pct:.1f}%) — kept as category")

print("\n✅ 'unknown' retained as a valid category in all columns")
print("   (No imputation — this is intentional)")


# ============================================================
# TASK 5 — ENCODE CATEGORICAL COLUMNS
#
# Strategy (from EDA):
#   Low/Medium cardinality  → One-Hot Encoding
#   High cardinality (>15)  → Target Encoding
#
# ⚠️ CRITICAL: Target Encoding MUST happen AFTER train/test
#    split to prevent data leakage. We compute encoding
#    statistics ONLY on train set, then apply to both.
#
# One-Hot columns (cardinality ≤ 15):
#   category(2), gender(3), belt_type(5), watch_type(9),
#   watch_style_category(9), style(12), case_material(13),
#   material(14), strap_material(11), brand(15)
#
# Target Encoding columns (cardinality > 15):
#   strap_color(37), color(67), dial_color(75)
# ============================================================

print("\n" + "─" * 60)
print("TASK 5 — Encoding Categorical Columns")
print("─" * 60)

# ── 5A. TRAIN / TEST SPLIT (before any encoding!) ──────────
# Split BEFORE encoding to avoid data leakage.
# We use log_price as target (created in Task 6),
# but for the split we use original price as proxy.

# Create log_price now so we can use it as split target
df["log_price"] = np.log1p(df["price"])

X = df.drop(columns=["price", "log_price"])
y_log = df["log_price"]
y_orig = df["price"]

X_train, X_test, y_train_log, y_test_log, y_train_orig, y_test_orig = train_test_split(
    X, y_log, y_orig, test_size=0.2, random_state=42
)

print(f"✅ Train/Test split: {len(X_train)} train | {len(X_test)} test (80/20)")

# ── 5B. ONE-HOT ENCODING ────────────────────────────────────
ohe_cols = [
    "category", "gender", "belt_type", "watch_type",
    "watch_style_category", "style", "case_material",
    "material", "strap_material", "brand"
]

# Only keep columns that actually exist
ohe_cols = [c for c in ohe_cols if c in X_train.columns]

print(f"\n   One-Hot Encoding {len(ohe_cols)} columns: {ohe_cols}")

X_train = pd.get_dummies(X_train, columns=ohe_cols, drop_first=False, dtype=int)
X_test  = pd.get_dummies(X_test,  columns=ohe_cols, drop_first=False, dtype=int)

# Align columns — test may be missing some dummy cols from train
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

print(f"   Shape after OHE — Train: {X_train.shape} | Test: {X_test.shape}")

# ── 5C. TARGET ENCODING ─────────────────────────────────────
# Formula: replace category value with mean(log_price) of
#          that category, computed ONLY on train set.
# Unknown categories in test → global mean (smoothing fallback)

target_enc_cols = ["strap_color", "color", "dial_color"]
target_enc_cols = [c for c in target_enc_cols if c in X_train.columns]

print(f"\n   Target Encoding {len(target_enc_cols)} high-cardinality columns: {target_enc_cols}")

global_mean = y_train_log.mean()
target_encoders = {}  # Store mappings so we can inspect/reuse

for col in target_enc_cols:
    # Compute mean log_price per category using TRAIN data only
    mean_map = y_train_log.groupby(X_train[col]).mean()
    target_encoders[col] = mean_map

    # Apply to train
    X_train[col] = X_train[col].map(mean_map).fillna(global_mean)

    # Apply to test (unseen categories → global mean)
    X_test[col]  = X_test[col].map(mean_map).fillna(global_mean)

    print(f"   {col}: {len(mean_map)} unique values encoded | global_mean={global_mean:.4f}")

print("✅ Target encoding complete (leakage-free)")


# ============================================================
# TASK 6 — LOG TRANSFORM TARGET
# Why: Price distribution is RIGHT SKEWED (EDA finding).
#      Log transform makes it more normal → better for
#      linear models + reduces effect of high-price outliers.
# np.log1p(x) = log(1+x) — safe for x=0 edge cases.
# At prediction time: np.expm1(pred) reverses the transform.
# ============================================================

print("\n" + "─" * 60)
print("TASK 6 — Log Transform Target (already created above)")
print("─" * 60)

print(f"   Original price  — mean: ₹{y_train_orig.mean():.0f} | skew: {y_train_orig.skew():.3f}")
print(f"   Log price       — mean: {y_train_log.mean():.4f}  | skew: {y_train_log.skew():.3f}")
print("✅ log_price = np.log1p(price)")
print("   To reverse at prediction: np.expm1(predicted_log_price)")


# ============================================================
# TASK 7 — SCALE NUMERICAL FEATURES
# Tree-based models (XGBoost, LightGBM, CatBoost, RF) do NOT
# need scaling. So we skip it for now.
# If you later use Linear Regression / Ridge / SVR → scale then.
# We note which columns would need scaling for reference.
# ============================================================

print("\n" + "─" * 60)
print("TASK 7 — Scaling Numerical Features")
print("─" * 60)

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
print(f"   Numerical columns ({len(numeric_cols)} total): {numeric_cols[:10]}...")
print("   ⚠️  Scaling SKIPPED — tree-based models don't need it.")
print("   ✅  For Linear/Ridge/SVR in Step 4, apply StandardScaler then.")


# ============================================================
# TASK 8 — SAVE OUTPUT
# We save 4 files:
#   watchvine_features_train_X.csv — Train features
#   watchvine_features_test_X.csv  — Test features
#   watchvine_features_train_y.csv — Train targets (log + orig)
#   watchvine_features_test_y.csv  — Test targets  (log + orig)
#
# This split is the canonical one for Steps 4–6.
# ============================================================

print("\n" + "─" * 60)
print("TASK 8 — Saving Output Files")
print("─" * 60)

# Combine targets into a single y dataframe
y_train_df = pd.DataFrame({"log_price": y_train_log.values, "price": y_train_orig.values},
                           index=X_train.index)
y_test_df  = pd.DataFrame({"log_price": y_test_log.values,  "price": y_test_orig.values},
                           index=X_test.index)

X_train.to_csv("watchvine_train_X.csv", index=False)
X_test.to_csv("watchvine_test_X.csv",   index=False)
y_train_df.to_csv("watchvine_train_y.csv", index=False)
y_test_df.to_csv("watchvine_test_y.csv",   index=False)

# Also save a combined full-features file for reference
df_full = X_train.copy()
df_full["log_price"] = y_train_log.values
df_full["price"]     = y_train_orig.values
df_full.to_csv("watchvine_features.csv", index=False)

print("✅ Saved: watchvine_train_X.csv")
print("✅ Saved: watchvine_test_X.csv")
print("✅ Saved: watchvine_train_y.csv")
print("✅ Saved: watchvine_test_y.csv")
print("✅ Saved: watchvine_features.csv  (train set combined — reference file)")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 3 COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Train set  : {X_train.shape[0]} rows × {X_train.shape[1]} features")
print(f"  Test set   : {X_test.shape[0]} rows × {X_test.shape[1]} features")
print(f"  Target     : log_price (reverse with np.expm1)")
print(f"  Dropped    : ai_is_automatic, ai_watch_type, name")
print(f"  Flags added: has_brand, has_details, has_watch_type")
print(f"  OHE cols   : {len(ohe_cols)} columns one-hot encoded")
print(f"  TE cols    : {len(target_enc_cols)} columns target encoded")
print(f"  Scaling    : Skipped (tree models don't need it)")
print("\n  ✅ Ready for Step 4 — Model Building!")
print("=" * 60)