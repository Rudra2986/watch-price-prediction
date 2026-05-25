# ============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# Project: Watch Price Prediction
# Author: Rudra
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# ── Plot Style ───────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 12

# ── Output folder for saving plots ──────────────────────────
os.makedirs("eda_plots", exist_ok=True)

# ── Load Cleaned Data ────────────────────────────────────────
df = pd.read_csv("watchvine_cleaned.csv")

print("✅ Data Loaded Successfully!")
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{list(df.columns)}")


# ============================================================
# 2.2 — BASIC INFO & OVERVIEW
# ============================================================

print("\n" + "="*60)
print("BASIC DATASET OVERVIEW")
print("="*60)

# Data types and non-null counts
print("\n📋 Data Types & Non-Null Counts:")
print(df.info())

# First few rows
print("\n👀 First 5 Rows:")
print(df.head())

# Statistical summary — numerical columns
print("\n📊 Numerical Summary:")
print(df.describe())

# Statistical summary — categorical columns
print("\n📊 Categorical Summary:")
print(df.describe(include='object'))

# Unique values per column
print("\n🔢 Unique Values Per Column:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique values")


# ============================================================
# 2.3 — TARGET VARIABLE ANALYSIS (price)
# ============================================================

print("\n" + "="*60)
print("TARGET VARIABLE ANALYSIS — price")
print("="*60)

# ── Basic Stats ──────────────────────────────────────────────
print("\n📊 Price Statistics:")
print(f"  Minimum Price  : ₹{df['price'].min():,.2f}")
print(f"  Maximum Price  : ₹{df['price'].max():,.2f}")
print(f"  Average Price  : ₹{df['price'].mean():,.2f}")
print(f"  Median Price   : ₹{df['price'].median():,.2f}")
print(f"  Std Deviation  : ₹{df['price'].std():,.2f}")
print(f"  Skewness       : {df['price'].skew():.4f}")
print(f"  Kurtosis       : {df['price'].kurt():.4f}")


# ── Plot 1: Distribution of Price ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram + KDE
sns.histplot(df['price'], bins=40, kde=True, color='steelblue', ax=axes[0])
axes[0].set_title('Price Distribution (Original)')
axes[0].set_xlabel('Price (₹)')
axes[0].set_ylabel('Count')

# Boxplot
sns.boxplot(y=df['price'], color='lightcoral', ax=axes[1])
axes[1].set_title('Price Boxplot (Outlier Check)')
axes[1].set_ylabel('Price (₹)')

plt.suptitle('Target Variable — Price Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/2_3_price_distribution.png', dpi=150)
plt.show()
print("✅ Plot saved: eda_plots/2_3_price_distribution.png")


# ── Outlier Detection using IQR ─────────────────────────────
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]

print(f"\n🔍 Outlier Detection (IQR Method):")
print(f"  Q1 (25%)       : ₹{Q1:,.2f}")
print(f"  Q3 (75%)       : ₹{Q3:,.2f}")
print(f"  IQR            : ₹{IQR:,.2f}")
print(f"  Lower Bound    : ₹{lower_bound:,.2f}")
print(f"  Upper Bound    : ₹{upper_bound:,.2f}")
print(f"  Outliers Found : {len(outliers)} rows")
print(f"  Outlier %      : {(len(outliers)/len(df))*100:.2f}%")


# ============================================================
# 2.4  BINARY FEATURE ANALYSIS (string-based)
# ============================================================

print("\n" + "="*60)
print("BINARY FEATURE ANALYSIS (is_automatic & ai_is_automatic)")
print("="*60)

# ── Value Counts ─────────────────────────────────────────────
binary_str_cols = ['is_automatic', 'ai_is_automatic']

for col in binary_str_cols:
    print(f"\n📌 {col} — Value Counts:")
    counts = df[col].value_counts()
    pct = df[col].value_counts(normalize=True) * 100
    for val in counts.index:
        print(f"   {val:10s} → {counts[val]:4d} rows  ({pct[val]:.1f}%)")

# ── Average Price per Category ───────────────────────────────
print(f"\n💰 Average Price by is_automatic:")
print(df.groupby('is_automatic')['price'].mean().sort_values(ascending=False).apply(lambda x: f"₹{x:,.2f}"))


# ── Plot 1: Value Distribution (Count) ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Countplot
order = ['Yes', 'No', 'Unknown']
sns.countplot(x=df['is_automatic'], order=order, 
              palette='Set2', ax=axes[0])
axes[0].set_title('is_automatic — Value Distribution')
axes[0].set_xlabel('Is Automatic?')
axes[0].set_ylabel('Count')

# Add count labels on bars
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}', 
                     (p.get_x() + p.get_width()/2, p.get_height()),
                     ha='center', va='bottom', fontweight='bold')

# Boxplot — price vs is_automatic
sns.boxplot(x=df['is_automatic'], y=df['price'],
            order=order, palette='Set2', ax=axes[1])
axes[1].set_title('is_automatic vs Price')
axes[1].set_xlabel('Is Automatic?')
axes[1].set_ylabel('Price (₹)')

plt.suptitle('is_automatic — Distribution & Price Impact',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/2_4_is_automatic_analysis.png', dpi=150)
plt.show()
print("✅ Plot saved: eda_plots/2_4_is_automatic_analysis.png")

# ── Similarity Check ─────────────────────────────────────────
print("\n🔍 Are is_automatic & ai_is_automatic identical?")
are_same = (df['is_automatic'] == df['ai_is_automatic']).all()
print(f"   Result: {'✅ YES — 100% identical (drop one in Step 3)' if are_same else '❌ NO — they differ'}")


# ============================================================
# 2.5 — CATEGORICAL FEATURE ANALYSIS
# ============================================================

print("\n" + "="*60)
print("CATEGORICAL FEATURE ANALYSIS")
print("="*60)

# ── Define categorical columns ───────────────────────────────
cat_cols = ['category', 'brand', 'color', 'style', 'material',
            'gender', 'dial_color', 'strap_material', 'strap_color',
            'ai_watch_type', 'case_material', 'belt_type',
            'watch_type', 'watch_style_category']

print(f"\n📋 Total Categorical Columns: {len(cat_cols)}")

# ── Cardinality Check (how many unique values each has) ──────
print("\n🔢 Cardinality (Unique Values per Column):")
print("-"*45)
for col in cat_cols:
    n = df[col].nunique()
    top = df[col].value_counts().index[0]
    top_pct = (df[col].value_counts().iloc[0] / len(df)) * 100
    print(f"  {col:25s} → {n:4d} unique  |  top: '{top}' ({top_pct:.1f}%)")

# ── Plot 1: Cardinality Bar Chart ────────────────────────────
cardinality = pd.Series(
    {col: df[col].nunique() for col in cat_cols}
).sort_values(ascending=False)

plt.figure(figsize=(14, 5))
bars = sns.barplot(x=cardinality.index, y=cardinality.values, 
                   palette='viridis')

# Add value labels on top of bars
for p in bars.patches:
    bars.annotate(f'{int(p.get_height())}',
                  (p.get_x() + p.get_width()/2, p.get_height()),
                  ha='center', va='bottom', fontweight='bold')

plt.title('Cardinality of Categorical Columns', 
          fontsize=14, fontweight='bold')
plt.xlabel('Column')
plt.ylabel('Number of Unique Values')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_plots/2_5_cardinality.png', dpi=150)
plt.show()
print("✅ Plot saved: eda_plots/2_5_cardinality.png")

# ── Plot 2: Top 10 Values for each categorical column ────────
print("\n📊 Generating distribution plots for each column...")

for col in cat_cols:
    top10 = df[col].value_counts().head(10)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Left: Count of top 10 values
    sns.barplot(x=top10.values, y=top10.index, 
                palette='Blues_r', ax=axes[0])
    axes[0].set_title(f'{col} — Top 10 Values (Count)')
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel(col)
    
    # Add count labels
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_width())}',
                         (p.get_width(), p.get_y() + p.get_height()/2),
                         ha='left', va='center', fontsize=9)

    # Right: Average price for top 10 values
    avg_price = (df[df[col].isin(top10.index)]
                 .groupby(col)['price']
                 .mean()
                 .reindex(top10.index))
    
    sns.barplot(x=avg_price.values, y=avg_price.index,
                palette='Oranges_r', ax=axes[1])
    axes[1].set_title(f'{col} — Avg Price (₹) per Category')
    axes[1].set_xlabel('Average Price (₹)')
    axes[1].set_ylabel('')

    # Add price labels
    for p in axes[1].patches:
        axes[1].annotate(f'₹{p.get_width():,.0f}',
                         (p.get_width(), p.get_y() + p.get_height()/2),
                         ha='left', va='center', fontsize=9)

    plt.suptitle(f'Analysis of: {col}', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'eda_plots/2_5_{col}.png', dpi=150)
    plt.show()
    print(f"✅ Saved: eda_plots/2_5_{col}.png")


# ── FIXED Unknown Summary ────────────────────────────────────
print("\n" + "-"*50)
print("'unknown' Value Summary per Column (Fixed)")
print("-"*50)

for col in cat_cols:
    unknown_count = df[col].str.lower().str.strip().eq('unknown').sum()
    unknown_pct = (unknown_count / len(df)) * 100
    status = "🔴 CRITICAL" if unknown_pct > 80 else "🟡 HIGH" if unknown_pct > 50 else "🟢 OK"
    if unknown_count > 0:
        print(f"  {col:25s} → {unknown_count:4d} unknowns ({unknown_pct:.1f}%) {status}")
    else:
        print(f"  {col:25s} → ✅ No unknowns")

# ── DIAGNOSE UNKNOWNS ────────────────────────────────────────
print("🔍 Checking all variations of 'unknown' in each column:\n")

for col in cat_cols:
    vals = df[col].value_counts()
    # Check ALL case variations
    for v in vals.index:
        if 'unknown' in str(v).lower():
            print(f"  {col:25s} → found: '{v}'  (count: {vals[v]})")

# ============================================================
# 2.6 — BRAND-WISE DEEP DIVE ANALYSIS
# ============================================================

print("\n" + "="*60)
print("BRAND-WISE DEEP DIVE ANALYSIS")
print("="*60)

# ── Basic Brand Stats ────────────────────────────────────────
total_brands = df['brand'].nunique()
print(f"\n📊 Total Unique Brands: {total_brands}")

brand_counts = df['brand'].value_counts()
print(f"\n🏆 Top Brands by Count:")
print(brand_counts)

# ── Brand Price Summary ──────────────────────────────────────
brand_price = df.groupby('brand')['price'].agg(
    Count='count',
    Mean_Price='mean',
    Median_Price='median',
    Min_Price='min',
    Max_Price='max',
    Std_Price='std'
).round(2).sort_values('Mean_Price', ascending=False)

print(f"\n💰 Brand Price Summary (sorted by Avg Price):")
print(brand_price.to_string())

# ── Separate unknown vs real brands ─────────────────────────
real_brands = df[df['brand'].str.lower() != 'unknown']
unknown_brands = df[df['brand'].str.lower() == 'unknown']

print(f"\n📊 Real Brand Watches    : {len(real_brands)} rows")
print(f"📊 Unknown Brand Watches : {len(unknown_brands)} rows")
print(f"📊 Avg price (real brand)   : ₹{real_brands['price'].mean():,.2f}")
print(f"📊 Avg price (unknown brand): ₹{unknown_brands['price'].mean():,.2f}")

# ── Plot 1: ALL Brands by Count (including unknown) ──────────
plt.figure(figsize=(14, 6))
bars = sns.barplot(x=brand_counts.values,
                   y=brand_counts.index,
                   palette='Blues_r')

for p in bars.patches:
    bars.annotate(f'{int(p.get_width())}',
                  (p.get_width(), p.get_y() + p.get_height()/2),
                  ha='left', va='center', fontsize=10,
                  fontweight='bold')

plt.title('All Brands — Number of Watches (incl. unknown)',
          fontsize=14, fontweight='bold')
plt.xlabel('Count')
plt.ylabel('Brand')
plt.tight_layout()
plt.savefig('eda_plots/2_6_brand_count.png', dpi=150)
plt.show()
print("✅ Plot saved: eda_plots/2_6_brand_count.png")

# ── Plot 2: Real Brands Only — Avg Price ─────────────────────
real_brand_price = brand_price[brand_price.index.str.lower() != 'unknown']

plt.figure(figsize=(14, 6))
bars = sns.barplot(x=real_brand_price['Mean_Price'].values,
                   y=real_brand_price.index,
                   palette='Oranges_r')

for p in bars.patches:
    bars.annotate(f'₹{p.get_width():,.0f}',
                  (p.get_width(), p.get_y() + p.get_height()/2),
                  ha='left', va='center', fontsize=10,
                  fontweight='bold')

plt.title('Real Brands Only — Average Price (₹)',
          fontsize=14, fontweight='bold')
plt.xlabel('Average Price (₹)')
plt.ylabel('Brand')
plt.tight_layout()
plt.savefig('eda_plots/2_6_brand_avg_price.png', dpi=150)
plt.show()
print("✅ Plot saved: eda_plots/2_6_brand_avg_price.png")


# ── Plot 3: Boxplot — Real Brands vs Price ───────────────────
real_brand_names = real_brand_price.index.tolist()
df_real = df[df['brand'].isin(real_brand_names)]

plt.figure(figsize=(14, 6))
sns.boxplot(x='brand', y='price',
            data=df_real,
            order=real_brand_names,
            palette='Set3')

plt.title('Real Brands — Price Distribution (Boxplot)',
          fontsize=14, fontweight='bold')
plt.xlabel('Brand')
plt.ylabel('Price (₹)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_plots/2_6_brand_boxplot.png', dpi=150)
plt.show()
print("✅ Plot saved: eda_plots/2_6_brand_boxplot.png")

# ── Plot 4: Real Brands — Mean Price ± Std Dev ───────────────
eligible = real_brand_price[real_brand_price['Count'] >= 5]

plt.figure(figsize=(14, 6))
plt.barh(eligible.index,
         eligible['Mean_Price'],
         xerr=eligible['Std_Price'],
         color='steelblue',
         alpha=0.7,
         error_kw={'ecolor': 'red', 'capsize': 5})

plt.title('Real Brands — Mean Price ± Std Deviation\n(brands with 5+ watches)',
          fontsize=14, fontweight='bold')
plt.xlabel('Price (₹)')
plt.ylabel('Brand')
plt.tight_layout()
plt.savefig('eda_plots/2_6_brand_price_spread.png', dpi=150)
plt.show()
print("✅ Plot saved: eda_plots/2_6_brand_price_spread.png")

# ── Brand Concentration ──────────────────────────────────────
print("\n📊 Brand Concentration (Real Brands Only):")
real_counts = brand_counts[brand_counts.index.str.lower() != 'unknown']
total_real  = real_counts.sum()

for n in [3, 5, len(real_counts)]:
    pct = (real_counts.head(n).sum() / len(df)) * 100
    print(f"  Top {n:2d} real brands → {real_counts.head(n).sum()} watches ({pct:.1f}% of full dataset)")

# ============================================================
# 2.7 — CORRELATION & FEATURE RELATIONSHIPS
# ============================================================

print("\n" + "="*60)
print("CORRELATION & FEATURE RELATIONSHIPS")
print("="*60)


# ── Part A: Encode categoricals temporarily for correlation ──
from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()
le = LabelEncoder()

for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Also encode binary string columns
for col in ['is_automatic', 'ai_is_automatic']:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

print("\n✅ Temporary label encoding done for correlation analysis")


# ── Part B: Correlation of every feature with PRICE ─────────
all_feature_cols = cat_cols + ['is_automatic', 'ai_is_automatic']

correlations = (df_encoded[all_feature_cols + ['price']]
                .corr()['price']
                .drop('price')
                .sort_values(ascending=False))

print("\n📊 Feature Correlation with Price:")
print("-"*55)
for col, val in correlations.items():
    bar = "█" * int(abs(val) * 40)
    direction = "↑" if val > 0 else "↓"
    print(f"  {col:25s} → {val:+.4f} {direction}  {bar}")\
    
# ── Plot 1: Correlation Bar Chart ────────────────────────────
colors = ['steelblue' if x > 0 else 'tomato' 
          for x in correlations.values]

plt.figure(figsize=(12, 7))
sns.barplot(x=correlations.values,
            y=correlations.index,
            palette=colors)

plt.axvline(x=0, color='black', linewidth=1.2)
plt.axvline(x=0.1,  color='green', linewidth=1,
            linestyle='--', alpha=0.5, label='±0.1 threshold')
plt.axvline(x=-0.1, color='green', linewidth=1,
            linestyle='--', alpha=0.5)

plt.title('Feature Correlation with Price\n(Label Encoded — Direction Indicator)',
          fontsize=13, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Feature')
plt.legend()
plt.tight_layout()
plt.savefig('eda_plots/2_7_correlation_bar.png', dpi=150)
plt.show()
print("✅ Plot saved: eda_plots/2_7_correlation_bar.png")

# ── Plot 2: Full Heatmap ─────────────────────────────────────
plt.figure(figsize=(16, 13))

corr_matrix = df_encoded[all_feature_cols + ['price']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            linewidths=0.5,
            annot_kws={'size': 7})

plt.title('Full Correlation Heatmap (All Features)',
          fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('eda_plots/2_7_full_heatmap.png', dpi=150)
plt.show()
print("✅ Plot saved: eda_plots/2_7_full_heatmap.png")

# ── Part C: Price Spread per Category ───────────────────────
print("\n📊 Price Spread per Categorical Column:")
print("-"*55)
print(f"  {'Column':25s}  {'Spread':>8}  {'Signal'}")
print("-"*55)

spread_results = {}

for col in cat_cols:
    mean_prices = df.groupby(col)['price'].mean()
    non_unknown = mean_prices[mean_prices.index.str.lower() != 'unknown']
    
    if len(non_unknown) >= 2:
        spread = non_unknown.max() - non_unknown.min()
        spread_results[col] = spread
        signal = "⭐⭐ Very Strong" if spread > 1000 else \
                 "⭐ Strong"       if spread > 500  else \
                 "〰 Moderate"    if spread > 200  else \
                 "✖ Weak"
        print(f"  {col:25s}  ₹{spread:>7,.0f}  {signal}")
    else:
        print(f"  {col:25s}  {'N/A':>8}  (not enough non-unknown values)")

# ── Plot 3: Price Spread Bar Chart ───────────────────────────
spread_series = pd.Series(spread_results).sort_values(ascending=False)

plt.figure(figsize=(13, 6))
colors = ['gold' if v > 1000 else
          'steelblue' if v > 500 else
          'lightblue' for v in spread_series.values]

bars = sns.barplot(x=spread_series.values,
                   y=spread_series.index,
                   palette=colors)

for p in bars.patches:
    bars.annotate(f'₹{p.get_width():,.0f}',
                  (p.get_width(), p.get_y() + p.get_height()/2),
                  ha='left', va='center', fontsize=9,
                  fontweight='bold')

plt.axvline(x=500, color='red', linestyle='--',
            linewidth=1.5, label='₹500 Strong Signal Threshold')
plt.axvline(x=1000, color='green', linestyle='--',
            linewidth=1.5, label='₹1000 Very Strong Threshold')

plt.title('Price Spread per Categorical Feature\n(max avg price − min avg price, excluding unknown)',
          fontsize=13, fontweight='bold')
plt.xlabel('Price Spread (₹)')
plt.ylabel('Feature')
plt.legend()
plt.tight_layout()
plt.savefig('eda_plots/2_7_price_spread.png', dpi=150)
plt.show()
print("✅ Plot saved: eda_plots/2_7_price_spread.png")

# ── Part D: Duplicate Column Check ──────────────────────────
print("\n" + "-"*55)
print("DUPLICATE COLUMN CHECK")
print("-"*55)

match = (df['is_automatic'] == df['ai_is_automatic']).sum()
total = len(df)
pct   = (match / total) * 100
print(f"\n  is_automatic == ai_is_automatic : {match}/{total} ({pct:.1f}%)")

if match == total:
    print("  🔴 VERDICT: 100% identical → DROP ai_is_automatic in Step 3")
else:
    diff = total - match
    print(f"  🟡 VERDICT: {diff} rows differ → Review before dropping")

# Check other potentially duplicate pairs
print(f"\n  watch_type vs ai_watch_type overlap check:")
overlap = df.groupby(['watch_type', 'ai_watch_type']).size().reset_index(name='count')
print(overlap.sort_values('count', ascending=False).head(8).to_string(index=False))


# ============================================================
# 2.8 — FEATURE QUALITY & CARDINALITY SUMMARY
# ============================================================

print("\n" + "="*60)
print("FEATURE QUALITY & CARDINALITY SUMMARY")
print("="*60)

# ── Build Master Summary Table ───────────────────────────────
summary_rows = []

for col in cat_cols:
    n_unique     = df[col].nunique()
    unknown_cnt  = (df[col].str.lower() == 'unknown').sum()
    unknown_pct  = (unknown_cnt / len(df)) * 100
    top_val      = df[col].value_counts().index[0]
    top_pct      = (df[col].value_counts().iloc[0] / len(df)) * 100

    # Cardinality label
    if n_unique <= 5:
        cardinality = 'Low'
    elif n_unique <= 20:
        cardinality = 'Medium'
    else:
        cardinality = 'High'

    # Quality label
    if unknown_pct == 0:
        quality = '🟢 Excellent'
    elif unknown_pct < 30:
        quality = '🟡 Good'
    elif unknown_pct < 60:
        quality = '🟠 Poor'
    else:
        quality = '🔴 Critical'

    summary_rows.append({
        'Column'       : col,
        'Unique'       : n_unique,
        'Cardinality'  : cardinality,
        'Unknown_Count': unknown_cnt,
        'Unknown_Pct'  : round(unknown_pct, 1),
        'Quality'      : quality,
        'Top_Value'    : top_val,
        'Top_Pct'      : round(top_pct, 1)
    })

summary_df = pd.DataFrame(summary_rows)
print("\n📋 Full Feature Quality Table:")
print(summary_df.to_string(index=False))


# ── Encoding Strategy Recommendation ────────────────────────
print("\n📋 Recommended Encoding Strategy for Step 3:")
print("-"*55)

for _, row in summary_df.iterrows():
    col   = row['Column']
    uniq  = row['Unique']
    upct  = row['Unknown_Pct']

    if uniq <= 5:
        strategy = 'One-Hot Encoding'
    elif uniq <= 15:
        strategy = 'One-Hot or Target Encoding'
    else:
        strategy = 'Target Encoding (high cardinality)'

    if upct > 80:
        note = '⚠ Create has_known flag too'
    elif upct > 50:
        note = '⚠ High unknowns — treat as valid category'
    else:
        note = '✅ Encode normally'

    print(f"  {col:25s} → {strategy:35s} | {note}")


 # ── Columns to DROP in Step 3 ────────────────────────────────
print("\n🗑 Columns Recommended to DROP in Step 3:")
print("-"*45)

drop_candidates = {
    'ai_is_automatic' : '100% identical to is_automatic',
    'ai_watch_type'   : 'Likely identical to watch_type — verify in 2.7',
    'name'            : 'Free text, too unique, not useful for ML'
}

for col, reason in drop_candidates.items():
    print(f"  ❌ {col:25s} → {reason}")

# ── Columns to CREATE in Step 3 (Feature Flags) ─────────────
print("\n✨ New Columns to CREATE in Step 3:")
print("-"*45)

create_candidates = {
    'has_brand'      : 'brand != unknown (1 if real brand, 0 if unknown)',
    'has_details'    : 'color/style/material != unknown (1 if info exists)',
    'has_watch_type' : 'watch_type != unknown (1 if type known)',
    'log_price'      : 'log(price) — for better model performance'
}

for col, reason in create_candidates.items():
    print(f"  ✅ {col:25s} → {reason}")


# ── Final Dataset Shape Summary ──────────────────────────────
print("\n" + "="*60)
print("FINAL DATASET SHAPE SUMMARY")
print("="*60)
print(f"\n  Total Rows          : {len(df):,}")
print(f"  Total Columns       : {len(df.columns)}")
print(f"  Numerical Columns   : 1 (price only)")
print(f"  Binary Str Columns  : 2 (is_automatic, ai_is_automatic)")
print(f"  Categorical Columns : {len(cat_cols)}")
print(f"  Target Variable     : price (₹{df['price'].min():,.0f} – ₹{df['price'].max():,.0f})")
print(f"  Fully Clean Columns : {(summary_df['Unknown_Pct'] == 0).sum()} columns")
print(f"  Critical Columns    : {(summary_df['Unknown_Pct'] > 80).sum()} columns (80%+ unknown)")

# ============================================================
# 2.9 — FINAL EDA INSIGHTS SUMMARY
# ============================================================

print("\n" + "="*60)
print("FINAL EDA INSIGHTS SUMMARY")
print("="*60)

insights = """
📌 INSIGHT 2.3 — TARGET VARIABLE (PRICE)
   • Price range: ₹1,999 – ₹7,499
   • Avg price: ~₹3,231 | Median slightly lower (right skewed)
   • 102 outliers (6.85%) — all on HIGH end (expensive watches)
   • Outliers are real data — do NOT remove
   • Log transformation recommended for Step 3

📌 INSIGHT 2.4 — BINARY FEATURES
   • is_automatic & ai_is_automatic are 100% identical
   • 89.1% values are 'unknown' in both columns
   • Drop ai_is_automatic in Step 3

📌 INSIGHT 2.5 — CATEGORICAL FEATURES
   • 14 categorical columns total
   • Only 'category' column has 0% unknowns
   • 853 watches (57.3%) have NO attribute details
   • 1,302 watches (87.5%) have unknown brand
   • color & dial_color have highest cardinality (67 & 75)

📌 INSIGHT 2.6 — BRAND ANALYSIS
   • Only 14 real brands in dataset
   • Only 186 watches (12.5%) have real brand info
   • Brand likely strong price signal despite high unknowns
   • Create has_brand flag in Step 3

📌 INSIGHT 2.7 — CORRELATIONS
   • Label correlation weak due to high unknown %
   • Price SPREAD more reliable signal for this dataset
   • Features with spread > ₹500 = strong model candidates
   • watch_type & ai_watch_type likely redundant → drop one
   • is_automatic & ai_is_automatic confirmed identical

📌 INSIGHT 2.8 — FEATURE QUALITY
   • Only category column is fully reliable (0% unknown)
   • Most columns are 🔴 Critical quality (57-89% unknown)
   • High cardinality columns (color, dial_color, strap_color)
     need Target Encoding in Step 3
   • Low cardinality columns (category, gender, belt_type)
     can use One-Hot Encoding
"""

print(insights)

# ── Save Insights to Text File ───────────────────────────────
with open('eda_insights.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("EDA INSIGHTS — Watch Price Prediction\n")
    f.write("="*60 + "\n")
    f.write(insights)

print("✅ Insights saved to: eda_insights.txt")
print("\n" + "="*60)
print("✅ STEP 2 — EDA COMPLETE!")
print("="*60)
print("\n🗂  All plots saved in: eda_plots/")
print("📄  Insights saved in : eda_insights.txt")
print("\n▶  Next → Step 3: Feature Engineering")
print("="*60)


