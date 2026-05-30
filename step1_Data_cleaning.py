# ============================================================
#  STEP 1 — DATA CLEANING
#  WatchVine Watch Price Prediction Project
# ============================================================

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. LOAD THE DATASET
# ─────────────────────────────────────────────
# low_memory=False stops pandas from guessing column types
# mid-read which causes errors on mixed-type columns.

df = pd.read_csv('watchvine_refined.products.csv', low_memory=False)

print("=" * 55)
print("ORIGINAL DATASET")
print("=" * 55)
print(f"Shape : {df.shape[0]} rows  x  {df.shape[1]} columns")


# ─────────────────────────────────────────────
# 2. DROP USELESS COLUMNS
# ─────────────────────────────────────────────
# We drop 3 groups of columns that carry zero
# predictive value for price:
#
# Group A — TEXT EMBEDDINGS (text_embedding[0..3071])
#   These are 3072 high-dimensional float vectors
#   generated from product text. Almost all rows
#   have them missing (only 16 rows have values).
#   Keeping 3072 near-empty columns is wasteful.
#
# Group B — IMAGE URLS (image_urls[0..10])
#   Raw image links. A model cannot use a URL
#   string to learn price — no signal here.
#
# Group C — METADATA / IDs
#   _id        : MongoDB document ID, not a feature
#   url        : Product page link, not a feature
#   scraped_at : Timestamp of web scraping, not useful
#   enhanced_at: Timestamp of AI enrichment, not useful
#   embedding_generated_at : Another timestamp
#   ai_analysis.analyzed_at: Another timestamp
#   ai_analysis.image_analyzed : True/False flag
#   ai_analysis.api_model : Which AI model analyzed it
#   category_key : Duplicate of category in slug form
#   searchable_text : Long text blob used for search,
#                     not structured for ML

cols_to_drop = (
    [c for c in df.columns if 'text_embedding' in c] +  # Group A
    [c for c in df.columns if 'image_urls'     in c] +  # Group B
    [                                                    # Group C
        '_id', 'url', 'scraped_at', 'enhanced_at',
        'embedding_generated_at', 'ai_analysis.analyzed_at',
        'ai_analysis.image_analyzed', 'ai_analysis.api_model',
        'category_key', 'searchable_text'
    ]
)

df.drop(columns=cols_to_drop, inplace=True)

print(f"\nAfter dropping useless columns : {df.shape[1]} columns remain")


# ─────────────────────────────────────────────
# 3. FIX THE TARGET VARIABLE — price
# ─────────────────────────────────────────────
# Problem 1: price is stored as a STRING, not a number.
#            e.g. '4099.00' instead of 4099.0
#
# Problem 2: 11 rows have the value 'Price Not Found'
#            which cannot be converted to a number.
#            These rows are useless for training because
#            we don't know what we're trying to predict.
#
# Fix: Use pd.to_numeric with errors='coerce'
#      → valid strings like '4099.00' become float 4099.0
#      → invalid strings like 'Price Not Found' become NaN
#      Then drop all rows where price is NaN.

df['price'] = pd.to_numeric(df['price'], errors='coerce')

before = len(df)
df.dropna(subset=['price'], inplace=True)
after  = len(df)

print(f"\nRows removed due to 'Price Not Found' : {before - after}")
print(f"Rows remaining                         : {after}")
print(f"Price dtype is now                     : {df['price'].dtype}")
print(f"Price range                            : ₹{df['price'].min()} – ₹{df['price'].max()}")


# ─────────────────────────────────────────────
# 4. DROP DATA LEAKAGE COLUMN — price_range
# ─────────────────────────────────────────────
# price_range contains values like:
#   "Mid-Range (₹1000-2500)", "Premium (₹2500-5000)", "Luxury (₹5000+)"
#
# This column was DIRECTLY DERIVED from the price.
# If we keep it, the model will use it to perfectly
# predict price without actually learning anything.
# This is called DATA LEAKAGE — the model cheats.
#
# Rule: Any column that contains information that
# would NOT be available at prediction time, OR is
# derived directly from the target → must be dropped.

df.drop(columns=['price_range'], inplace=True)
print("\nDropped 'price_range' → prevents data leakage")


# ─────────────────────────────────────────────
# 5. KEEP ONLY THE BEST COLOR / STYLE / MATERIAL
# ─────────────────────────────────────────────
# The dataset has colors[0], colors[1] ... colors[4]
# Similarly for styles and materials.
#
# colors[0] = primary color  → most filled, most useful
# colors[1], colors[2] etc.  → very sparse (1016-1498 missing)
#
# Strategy: Keep only [0] columns (primary attributes)
#           and drop the rest to reduce noise.

cols_to_drop_multi = (
    [c for c in df.columns if 'colors['    in c and '[0]' not in c] +
    [c for c in df.columns if 'styles['    in c and '[0]' not in c] +
    [c for c in df.columns if 'materials[' in c and '[0]' not in c] +
    [c for c in df.columns if 'design_elements' in c]   # very sparse
)

df.drop(columns=cols_to_drop_multi, inplace=True)
print(f"\nKept only primary color/style/material columns")
print(f"Columns now : {df.shape[1]}")


# ─────────────────────────────────────────────
# 6. RENAME COLUMNS FOR CLARITY
# ─────────────────────────────────────────────
# Column names like 'ai_analysis.additional_details.dial_color'
# are too long and hard to work with. We rename them
# to short, clean names.

df.rename(columns={
    'colors[0]'                                          : 'color',
    'styles[0]'                                          : 'style',
    'materials[0]'                                       : 'material',
    'ai_analysis.additional_details.dial_color'          : 'dial_color',
    'ai_analysis.additional_details.strap_material'      : 'strap_material',
    'ai_analysis.additional_details.strap_color'         : 'strap_color',
    'ai_analysis.additional_details.watch_type'          : 'ai_watch_type',
    'ai_analysis.additional_details.case_material'       : 'case_material',
    'ai_analysis.additional_details.is_automatic'        : 'ai_is_automatic',
    'ai_analysis.additional_details.watch_style_category': 'watch_style_category',
}, inplace=True)

print("\nColumns renamed for clarity ✓")


# ─────────────────────────────────────────────
# 7. HANDLE MISSING VALUES & FEATURE EXTRACTION
# ─────────────────────────────────────────────
# Different strategies for different column types:
#
# CATEGORICAL columns → fill with 'Unknown'
#   Reason: 'Unknown' is itself a valid category.
#   However, brand is highly missing (87.5%).
#   We first extract the brand from the name column if it is missing.
#
# BOOLEAN columns (is_automatic) → fill with 'Unknown'
#   We also extract automatic movement status from the name column if missing.

def extract_brand(name):
    if pd.isna(name):
        return 'Unknown'
    name_lower = str(name).lower().strip()
    words = name_lower.split()
    if not words:
        return 'Unknown'
    first_word = words[0]
    
    if 'fossi_l' in first_word or 'fossil' in first_word:
        return 'Fossil'
    elif 'role_x' in first_word or 'rolex' in first_word:
        return 'Rolex'
    elif 'casio' in first_word:
        return 'Casio'
    elif 'rad_o' in first_word or 'rado' in first_word:
        return 'Rado'
    elif 'emporio' in first_word or 'arman_i' in first_word or 'armani' in first_word:
        return 'Armani'
    elif 'tisso_t' in first_word or 'tissot' in first_word:
        return 'Tissot'
    elif 'omeg_a' in first_word or 'omega' in first_word:
        return 'Omega'
    elif 'hublot' in first_word or 'hublo_t' in first_word:
        return 'Hublot'
    elif 'seiko' in first_word:
        return 'Seiko'
    elif 'audemars' in first_word:
        return 'Audemars Piguet'
    elif 'tommy' in first_word or 'tomm_y' in first_word:
        return 'Tommy Hilfiger'
    elif 'patek' in first_word or 'pate_k' in first_word:
        return 'Patek Philippe'
    elif 'diese_l' in first_word or 'diesel' in first_word:
        return 'Diesel'
    elif 'guess' in first_word or 'gues_s' in first_word:
        return 'Guess'
    elif 'cartie_r' in first_word or 'cartier' in first_word:
        return 'Cartier'
    elif 'richard' in first_word:
        return 'Richard Mille'
    elif 'tag' in first_word:
        return 'Tag Heuer'
    elif 'nik_e' in first_word or 'nike' in first_word:
        return 'Nike'
    elif 'maserati' in first_word or 'maserat_i' in first_word:
        return 'Maserati'
    elif 'gucc_i' in first_word or 'gucci' in first_word:
        return 'Gucci'
    elif 'chane_l' in first_word or 'chanel' in first_word:
        return 'Chanel'
    elif 'versace' in first_word:
        return 'Versace'
    elif 'jacob' in first_word:
        return 'Jacob & Co'
    elif 'iwc' in first_word:
        return 'IWC'
    elif 'bvlgari' in first_word or 'bulgari' in first_word:
        return 'Bvlgari'
    elif 'michael_kors' in first_word or 'kors' in first_word:
        return 'Michael Kors'
    elif 'corum' in first_word:
        return 'Corum'
    elif 'oakle_y' in first_word or 'oakley' in first_word:
        return 'Oakley'
    elif 'boss' in first_word:
        return 'Hugo Boss'
    elif 'loui_s' in first_word or 'vuitton' in first_word:
        return 'Louis Vuitton'
    elif 'citizen' in first_word:
        return 'Citizen'
    elif 'maxima' in first_word:
        return 'Maxima'
    else:
        for b in ['fossil', 'rolex', 'casio', 'rado', 'armani', 'tissot', 'omega', 'hublot', 'seiko', 'audemars', 'tommy', 'patek', 'diesel', 'guess', 'cartier', 'richard', 'tag', 'nike', 'maserati', 'gucci', 'chanel', 'versace', 'jacob', 'iwc', 'bvlgari', 'michael kors', 'corum', 'oakley', 'boss', 'louis vuitton', 'citizen', 'maxima']:
            if b in name_lower:
                if b == 'audemars': return 'Audemars Piguet'
                if b == 'tommy': return 'Tommy Hilfiger'
                if b == 'patek': return 'Patek Philippe'
                if b == 'richard': return 'Richard Mille'
                if b == 'tag': return 'Tag Heuer'
                return b.title()
        if len(first_word) > 2 and first_word not in ['mens', 'ladies', 'boys', 'girls', 'watch', 'watches', 'classic', 'new', 'luxury', 'analog', 'digital']:
            return first_word.title()
        return 'Unknown'

# Extract and fill brand
df['extracted_brand'] = df['name'].apply(extract_brand)
df['brand'] = df['brand'].fillna(df['extracted_brand'])
df.drop(columns=['extracted_brand'], inplace=True, errors='ignore')

categorical_cols = [
    'brand', 'color', 'style', 'material', 'gender',
    'belt_type', 'watch_type', 'dial_color',
    'strap_material', 'strap_color', 'ai_watch_type',
    'case_material', 'watch_style_category'
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Clean is_automatic checking both current status and name keywords
def clean_is_automatic_with_name(row, col_name):
    val = row[col_name]
    name = str(row['name']).lower()
    
    if not pd.isna(val):
        v = str(val).strip().lower()
        if v in ('true', '1', 'yes'):
            return 'Yes'
        if v in ('false', '0', 'no'):
            return 'No'
            
    if any(keyword in name for keyword in ['automatic', 'meccanico', 'skeleton', 'self-winding']):
        return 'Yes'
        
    return 'Unknown'

for col in ['is_automatic', 'ai_is_automatic']:
    if col in df.columns:
        df[col] = df.apply(lambda r: clean_is_automatic_with_name(r, col), axis=1)

print("\nMissing values handled & feature extraction complete ✓")
print(f"Missing values remaining :\n{df.isnull().sum()[df.isnull().sum() > 0]}")


# ─────────────────────────────────────────────
# 8. STANDARDIZE TEXT VALUES
# ─────────────────────────────────────────────
# Many text columns have inconsistent casing.
# e.g. 'rubber', 'Rubber', 'RUBBER' are the same.
# We convert everything to lowercase + strip spaces
# so they're treated as one category, not three.

text_cols = categorical_cols + ['category']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()

print("\nText values standardized (lowercase + stripped) ✓")


# ─────────────────────────────────────────────
# 9. FINAL CHECK
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("CLEANED DATASET SUMMARY")
print("=" * 55)
print(f"Shape      : {df.shape[0]} rows  x  {df.shape[1]} columns")
print(f"\nColumns    :\n{df.columns.tolist()}")
print(f"\nData Types :\n{df.dtypes}")
print(f"\nMissing Values :\n{df.isnull().sum()}")
print(f"\nPrice Stats :\n{df['price'].describe()}")
print(f"\nSample rows :")
print(df[['name','brand','color','style','material',
          'belt_type','watch_type','gender','price']].head(5).to_string())


# ─────────────────────────────────────────────
# 10. SAVE THE CLEANED DATASET
# ─────────────────────────────────────────────
# Save so we can directly load it in Step 2
# without repeating all the cleaning.

df.to_csv('watchvine_cleaned.csv', index=False)
print("\n✅ Cleaned dataset saved as 'watchvine_cleaned.csv'")