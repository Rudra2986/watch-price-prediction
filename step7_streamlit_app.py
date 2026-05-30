# ============================================================
# STEP 7 — STREAMLIT DEPLOYMENT (v2 — Premium UI)
# Project : Watch Price Prediction (WatchVine Dataset)
# Run     : streamlit run step7_streamlit_app.py
# Author  : Rudra (GitHub: Rudra2986)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="WatchVine Price Predictor",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ──────────────────────────────── */
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* ── Header ──────────────────────────────── */
    .hero {
        text-align: center;
        padding: 30px 0 10px 0;
    }
    .hero-icon {
        font-size: 3.5rem;
        margin-bottom: 5px;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero p {
        color: rgba(255,255,255,0.5);
        font-size: 1rem;
        margin-top: 4px;
        font-weight: 300;
    }
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.4), transparent);
        margin: 15px 0 25px 0;
    }

    /* ── Section Labels ──────────────────────── */
    .section-label {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label span {
        font-size: 1.2rem;
    }

    /* ── Cards ────────────────────────────────── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
    }

    /* ── Result ───────────────────────────────── */
    .result-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.2));
        backdrop-filter: blur(30px);
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        margin: 20px 0;
        animation: fadeIn 0.6s ease-out;
    }
    .result-card .label {
        color: rgba(255,255,255,0.6);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    .result-card .price {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c4b5fd, #a78bfa, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .result-card .range {
        color: rgba(255,255,255,0.45);
        font-size: 0.95rem;
        margin-top: 8px;
        font-weight: 400;
    }

    /* ── Metric Boxes ────────────────────────── */
    .metric-row {
        display: flex;
        gap: 15px;
        margin-top: 15px;
    }
    .metric-box {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-box .val {
        font-size: 1.3rem;
        font-weight: 700;
        color: #c4b5fd;
    }
    .metric-box .lbl {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.4);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Info Badge ───────────────────────────── */
    .info-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(139,92,246,0.15);
        border: 1px solid rgba(139,92,246,0.25);
        border-radius: 8px;
        padding: 6px 12px;
        color: #c4b5fd;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 3px;
    }

    /* ── Footer ───────────────────────────────── */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.25);
        font-size: 0.8rem;
        padding: 30px 0 10px 0;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 40px;
    }
    .footer a {
        color: #a78bfa;
        text-decoration: none;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Streamlit Overrides ──────────────────── */
    .stSelectbox label, .stRadio label {
        color: rgba(255,255,255,0.7) !important;
        font-weight: 500 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.35) !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL & ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("final_model.pkl")
    te_data = joblib.load("target_encoders.pkl")
    train_cols = joblib.load("train_columns.pkl")
    metadata = joblib.load("model_metadata.pkl") if os.path.exists("model_metadata.pkl") else {}
    return model, te_data, train_cols, metadata

try:
    model, te_data, train_cols, metadata = load_artifacts()
    target_encoders = te_data["encoders"]
    global_mean = te_data["global_mean"]
except FileNotFoundError as e:
    st.error(f"Missing artifact: {e}. Please run Steps 3-6 first.")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-icon">⌚</div>
    <h1>WatchVine Price Predictor</h1>
    <p>AI-powered watch price estimation using machine learning</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — MODEL INFO
# ============================================================

model_name = metadata.get("model_name", "CatBoost")
r2_val = metadata.get("r2", 0)
rmse_val = metadata.get("rmse_rs", 0)

with st.sidebar:
    st.markdown("### ⚙️ Model Details")
    st.markdown(f"""
    <div class="glass-card">
        <div class="info-badge">🤖 {model_name}</div>
        <div class="info-badge">📊 R² = {r2_val:.4f}</div>
        <div class="info-badge">📏 RMSE = ₹{rmse_val:.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 About")
    st.markdown("""
    Predicts watch prices from the **WatchVine** Indian
    e-commerce dataset using a tuned **CatBoost** model.

    **Training data**: 1,190 watches
    **Features**: 96 engineered features
    **Target**: log-transformed price
    """)

    st.markdown("---")
    st.markdown("[🔗 GitHub](https://github.com/Rudra2986/watch-price-prediction) · Built by **Rudra**")


# ============================================================
# INPUT FORM — 10 Simplified Inputs
# ============================================================
# Removed 4 redundant inputs:
#   gender            → auto-derived from category
#   watch_style_cat   → 100% identical to watch_type
#   color             → 95.6% identical to dial_color
#   material          → 97.5% identical to strap_material
# ============================================================

st.markdown('<div class="section-label"><span>📝</span> Watch Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    category = st.selectbox("🏷️ Category", [
        "Men's Watch", "Women's Watch"
    ])

    brand = st.selectbox("🏢 Brand", [
        "Unknown", "Casio", "Seiko", "Citizen", "Tommy Hilfiger",
        "Michael Kors", "Maxima", "Tag Heuer", "Versace", "Hublot",
        "IWC", "Franck Muller", "Audemars Piguet", "Patek Philippe",
        "Richard Mille"
    ])

    watch_type = st.selectbox("⌚ Watch Type", [
        "Unknown", "Casual", "Sports", "Fashion", "Dress",
        "Luxury", "Diving", "Professional", "Racing"
    ])

    style = st.selectbox("🎨 Style", [
        "Unknown", "Analog", "Digital", "Analog-Digital",
        "Smartwatch", "Sporty", "Formal", "Luxury",
        "Minimalistic", "Vintage", "Hybrid"
    ])

    is_automatic = st.radio("⚙️ Movement Type", [
        "Unknown", "Automatic", "Non-Automatic"
    ], horizontal=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    belt_type = st.selectbox("🔗 Belt Type", [
        "Unknown", "Leather Belt", "Metal Belt",
        "Rubber Belt", "Other"
    ])

    case_material = st.selectbox("🛡️ Case Material", [
        "Unknown", "Stainless Steel", "Gold", "Rose Gold",
        "Titanium", "Ceramic", "Plastic", "Metal",
        "Carbon Fiber"
    ])

    strap_material = st.selectbox("📿 Strap Material", [
        "Unknown", "Leather", "Stainless Steel", "Metal",
        "Rubber", "Silicone", "Fabric", "Ceramic", "Plastic"
    ])

    dial_color = st.selectbox("🎯 Dial Color", [
        "Unknown", "Black", "White", "Blue", "Silver",
        "Gold", "Green", "Brown", "Grey", "Rose Gold",
        "Red", "Navy", "Skeleton", "Champagne", "Pink"
    ])

    strap_color = st.selectbox("🎨 Strap Color", [
        "Unknown", "Black", "Silver", "Brown", "Gold",
        "Blue", "Rose Gold", "Grey", "White", "Green",
        "Two-Tone", "Red", "Navy"
    ])

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PREPROCESSING — Replicate Step 3 Exactly
# ============================================================

def preprocess_input(category, brand, watch_type, style, is_automatic,
                     belt_type, case_material, strap_material,
                     dial_color, strap_color):
    """
    Replicate Step 3 transformations.
    Auto-derives 4 redundant fields from user inputs:
      gender            ← category
      watch_style_cat   ← watch_type (100% identical)
      color             ← dial_color (95.6% identical)
      material          ← strap_material (97.5% identical)
    """

    # ── Normalize to lowercase (Step 1 lowercased everything) ──
    cat_lower = category.lower()
    brand_lower = brand.lower()
    wtype_lower = watch_type.lower()
    style_lower = style.lower().replace("non-automatic", "unknown")
    belt_lower = belt_type.lower().replace(" ", "_")
    case_lower = case_material.lower()
    strap_mat_lower = strap_material.lower()
    dial_lower = dial_color.lower()
    strap_c_lower = strap_color.lower()

    # Map style names that differ slightly
    style_map = {
        "analog-digital": "analog-digital",
        "digital": "unknown",  # 'digital' not in training OHE columns
    }
    style_lower = style_map.get(style_lower, style_lower)

    # Map case materials with training-compatible names
    case_map = {
        "carbon fiber": "carbon fiber",
    }
    case_lower = case_map.get(case_lower, case_lower)

    # ── Auto-derive redundant fields ─────────────────────────
    # gender from category
    if cat_lower == "men's watch":
        gender = "men"
    elif cat_lower == "women's watch":
        gender = "women"
    else:
        gender = "unknown"

    # watch_style_category = watch_type (100% identical in dataset)
    watch_style_cat = wtype_lower

    # color = dial_color (95.6% identical)
    color = dial_lower

    # material = strap_material (97.5% identical)
    material = strap_mat_lower

    # ── is_automatic encoding ────────────────────────────────
    auto_map = {"automatic": 1, "non-automatic": 0, "unknown": -1}
    is_auto_val = auto_map.get(is_automatic.lower(), -1)

    # ── Flag columns ─────────────────────────────────────────
    has_brand = 1 if brand_lower != "unknown" else 0
    has_details = 1 if color != "unknown" else 0
    has_watch_type = 1 if wtype_lower != "unknown" else 0

    # ── Target encode high-cardinality columns ───────────────
    def safe_encode(col_name, value):
        encoder = target_encoders.get(col_name, {})
        if hasattr(encoder, 'get'):
            result = encoder.get(value, global_mean)
        else:
            result = global_mean
        # Handle edge cases
        if hasattr(result, '__iter__') and not isinstance(result, str):
            return global_mean
        return float(result) if result is not None else global_mean

    color_enc = safe_encode("color", color)
    dial_enc = safe_encode("dial_color", dial_lower)
    strap_c_enc = safe_encode("strap_color", strap_c_lower)

    # ── Build feature row ────────────────────────────────────
    row = {col: 0 for col in train_cols}  # Initialize all to 0

    # Numeric / encoded columns
    row["color"] = color_enc
    row["dial_color"] = dial_enc
    row["strap_color"] = strap_c_enc
    row["is_automatic"] = is_auto_val
    row["has_brand"] = has_brand
    row["has_details"] = has_details
    row["has_watch_type"] = has_watch_type

    # One-hot encode by setting the correct column to 1
    ohe_mappings = {
        "category": cat_lower,
        "gender": gender,
        "belt_type": belt_lower,
        "watch_type": wtype_lower,
        "watch_style_category": watch_style_cat,
        "style": style_lower,
        "case_material": case_lower,
        "material": material,
        "strap_material": strap_mat_lower,
        "brand": brand_lower,
    }

    for prefix, value in ohe_mappings.items():
        ohe_col = f"{prefix}_{value}"
        if ohe_col in row:
            row[ohe_col] = 1

    return pd.DataFrame([row])[train_cols]


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

predict_clicked = st.button("⚡ Predict Price", use_container_width=True, type="primary")

if predict_clicked:
    input_df = preprocess_input(
        category, brand, watch_type, style, is_automatic,
        belt_type, case_material, strap_material,
        dial_color, strap_color
    )

    log_pred = model.predict(input_df)[0]
    predicted_price = np.expm1(log_pred)
    rmse = metadata.get("rmse_rs", 500)
    price_low = max(predicted_price - rmse, 999)
    price_high = predicted_price + rmse

    # ── Result Card ──────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card">
        <div class="label">Estimated Price</div>
        <div class="price">₹{predicted_price:,.0f}</div>
        <div class="range">Confidence range: ₹{price_low:,.0f} — ₹{price_high:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Boxes ─────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="val">₹{predicted_price:,.0f}</div>
            <div class="lbl">Point Estimate</div>
        </div>
        <div class="metric-box">
            <div class="val">₹{price_low:,.0f} — ₹{price_high:,.0f}</div>
            <div class="lbl">Confidence Range</div>
        </div>
        <div class="metric-box">
            <div class="val">{log_pred:.4f}</div>
            <div class="lbl">Log Price</div>
        </div>
        <div class="metric-box">
            <div class="val">{r2_val:.2%}</div>
            <div class="lbl">Model Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Summary ────────────────────────────────────────
    with st.expander("📋 Your Input Summary"):
        summary = {
            "Category": category, "Brand": brand,
            "Watch Type": watch_type, "Style": style,
            "Movement": is_automatic, "Belt Type": belt_type,
            "Case Material": case_material, "Strap Material": strap_material,
            "Dial Color": dial_color, "Strap Color": strap_color,
        }
        sc1, sc2 = st.columns(2)
        items = list(summary.items())
        for k, v in items[:5]:
            sc1.markdown(f"**{k}:** {v}")
        for k, v in items[5:]:
            sc2.markdown(f"**{k}:** {v}")

    # ── SHAP Feature Importance ──────────────────────────────
    with st.expander("🔍 What's driving this prediction? (SHAP)"):
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            shap_df = pd.DataFrame({
                "Feature": train_cols,
                "Impact": shap_values[0]
            })
            shap_df["Abs"] = shap_df["Impact"].abs()
            top_feats = shap_df.nlargest(12, "Abs").sort_values("Impact", ascending=True)

            st.bar_chart(
                top_feats.set_index("Feature")["Impact"],
                use_container_width=True,
                color="#a78bfa"
            )
            st.caption("↑ Positive = pushes price UP · Negative = pushes price DOWN")
        except Exception as e:
            st.info(f"SHAP analysis unavailable: {e}")

else:
    # ── Empty State ──────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:40px 0; color:rgba(255,255,255,0.3);">
        <p style="font-size:2.5rem; margin-bottom:10px;">⌚</p>
        <p style="font-size:1.1rem;">Select watch details above and click <strong>Predict Price</strong></p>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    WatchVine Price Predictor · Summer Internship Project ·
    <a href="https://github.com/Rudra2986/watch-price-prediction">GitHub</a>
    · Built by Rudra
</div>
""", unsafe_allow_html=True)
