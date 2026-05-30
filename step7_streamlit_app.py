# ============================================================
# STEP 7 — STREAMLIT DEPLOYMENT (v3 — Premium AI Product)
# Project : Watch Price Prediction (WatchVine Dataset)
# Run     : streamlit run step7_streamlit_app.py
# Author  : Rudra (GitHub: Rudra2986)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Chronos AI — Watch Price Intelligence",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PREMIUM CSS — Futuristic AI Dashboard
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #06060f;
        --bg-card: rgba(255,255,255,0.025);
        --bg-card-hover: rgba(255,255,255,0.05);
        --border: rgba(255,255,255,0.06);
        --border-glow: rgba(139,92,246,0.3);
        --accent: #8b5cf6;
        --accent-2: #06b6d4;
        --accent-3: #f59e0b;
        --text-primary: rgba(255,255,255,0.92);
        --text-secondary: rgba(255,255,255,0.55);
        --text-muted: rgba(255,255,255,0.3);
        --gradient-1: linear-gradient(135deg, #8b5cf6, #06b6d4);
        --gradient-2: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        --glow-purple: 0 0 30px rgba(139,92,246,0.15);
        --glow-cyan: 0 0 30px rgba(6,182,212,0.15);
    }

    /* ── Global ──────────────────────────────── */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] { display: none; }
    .block-container { max-width: 1200px; padding-top: 1rem; }
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Floating Orbs (Background) ──────────── */
    .orb-container {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: 0; overflow: hidden;
    }
    .orb {
        position: absolute; border-radius: 50%; filter: blur(80px);
        animation: float 20s ease-in-out infinite;
    }
    .orb-1 {
        width: 500px; height: 500px; top: -10%; left: -5%;
        background: rgba(139,92,246,0.08);
        animation-delay: 0s;
    }
    .orb-2 {
        width: 400px; height: 400px; bottom: -10%; right: -5%;
        background: rgba(6,182,212,0.06);
        animation-delay: -7s;
    }
    .orb-3 {
        width: 300px; height: 300px; top: 40%; left: 50%;
        background: rgba(236,72,153,0.04);
        animation-delay: -14s;
    }
    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(30px, -40px) scale(1.05); }
        66% { transform: translate(-20px, 30px) scale(0.95); }
    }

    /* ── Hero ────────────────────────────────── */
    .hero-section {
        text-align: center;
        padding: 50px 0 20px 0;
        position: relative;
    }
    .brand-badge {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(139,92,246,0.1);
        border: 1px solid rgba(139,92,246,0.2);
        border-radius: 100px; padding: 6px 18px;
        font-size: 0.78rem; color: #a78bfa;
        font-weight: 500; letter-spacing: 1.5px;
        text-transform: uppercase; margin-bottom: 20px;
        animation: fadeDown 0.8s ease-out;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem; font-weight: 700;
        line-height: 1.1; margin: 0;
        background: linear-gradient(135deg, #e0e7ff 0%, #a78bfa 40%, #06b6d4 70%, #e0e7ff 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeDown 0.8s ease-out, shimmer 6s ease-in-out infinite;
    }
    .hero-sub {
        color: var(--text-secondary); font-size: 1.05rem;
        margin-top: 12px; font-weight: 300;
        animation: fadeDown 0.8s ease-out 0.2s both;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    @keyframes fadeDown {
        from { opacity: 0; transform: translateY(-15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Navigation Tabs ─────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; background: rgba(255,255,255,0.03);
        border-radius: 14px; padding: 4px;
        border: 1px solid var(--border);
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 10px 28px;
        color: var(--text-secondary); font-weight: 500;
        font-size: 0.88rem; border: none;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(139,92,246,0.15) !important;
        color: #c4b5fd !important;
        border: 1px solid rgba(139,92,246,0.25) !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 25px; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    /* ── Glass Card ───────────────────────────── */
    .g-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 18px; padding: 28px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeUp 0.6s ease-out both;
    }
    .g-card:hover {
        background: var(--bg-card-hover);
        border-color: rgba(139,92,246,0.15);
        box-shadow: var(--glow-purple);
        transform: translateY(-2px);
    }
    .card-label {
        font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 1.8px; color: var(--text-muted);
        font-weight: 600; margin-bottom: 8px;
    }

    /* ── KPI Cards ────────────────────────────── */
    .kpi-row { display: flex; gap: 16px; margin: 20px 0; }
    .kpi-card {
        flex: 1; background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px; padding: 22px;
        text-align: center;
        transition: all 0.4s ease;
        animation: fadeUp 0.6s ease-out both;
    }
    .kpi-card:hover {
        border-color: rgba(139,92,246,0.2);
        box-shadow: var(--glow-purple);
        transform: translateY(-3px);
    }
    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem; font-weight: 700;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label {
        font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 1.5px; color: var(--text-muted);
        font-weight: 500; margin-top: 6px;
    }
    .kpi-icon { font-size: 1.5rem; margin-bottom: 8px; }

    /* ── Result Card ──────────────────────────── */
    .result-container {
        background: linear-gradient(135deg, rgba(139,92,246,0.08), rgba(6,182,212,0.05));
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 24px; padding: 45px 30px;
        text-align: center; margin: 25px 0;
        position: relative; overflow: hidden;
        animation: fadeUp 0.5s ease-out;
    }
    .result-container::before {
        content: ''; position: absolute;
        top: 0; left: 50%; transform: translateX(-50%);
        width: 60%; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.5), transparent);
    }
    .result-label {
        font-size: 0.78rem; text-transform: uppercase;
        letter-spacing: 3px; color: var(--text-muted);
        font-weight: 600; margin-bottom: 10px;
    }
    .result-price {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem; font-weight: 700;
        background: linear-gradient(135deg, #e0e7ff, #a78bfa, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1; margin: 5px 0;
        animation: priceReveal 0.8s ease-out;
    }
    .result-range {
        color: var(--text-secondary);
        font-size: 0.92rem; font-weight: 400;
        margin-top: 8px;
    }
    .result-range span {
        color: #a78bfa; font-weight: 600;
    }
    @keyframes priceReveal {
        from { opacity: 0; transform: scale(0.8); filter: blur(10px); }
        to { opacity: 1; transform: scale(1); filter: blur(0); }
    }

    /* ── Metrics Row ──────────────────────────── */
    .metrics-row { display: flex; gap: 14px; margin: 20px 0; }
    .metric-pill {
        flex: 1; background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px; padding: 18px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-pill:hover {
        border-color: rgba(139,92,246,0.2);
    }
    .metric-pill .val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem; font-weight: 600;
        color: var(--text-primary);
    }
    .metric-pill .lbl {
        font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 1.5px; color: var(--text-muted);
        margin-top: 5px; font-weight: 500;
    }

    /* ── Insight Card ─────────────────────────── */
    .insight-card {
        background: rgba(139,92,246,0.06);
        border: 1px solid rgba(139,92,246,0.12);
        border-radius: 14px; padding: 18px 22px;
        margin: 10px 0; display: flex;
        align-items: flex-start; gap: 14px;
        animation: fadeUp 0.6s ease-out both;
    }
    .insight-card .icon {
        font-size: 1.4rem; margin-top: 2px;
    }
    .insight-card .text {
        color: var(--text-secondary);
        font-size: 0.88rem; line-height: 1.6;
    }
    .insight-card .text strong {
        color: #c4b5fd;
    }

    /* ── Buttons ──────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #6366f1, #0ea5e9) !important;
        background-size: 200% 200% !important;
        color: white !important; border: none !important;
        border-radius: 14px !important;
        padding: 16px 32px !important;
        font-weight: 600 !important; font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.25) !important;
    }
    .stButton > button:hover {
        background-position: 100% 50% !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 35px rgba(99,102,241,0.4) !important;
    }
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }

    /* ── Select Boxes ─────────────────────────── */
    .stSelectbox label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important; font-size: 0.85rem !important;
    }
    .stRadio label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    div[data-testid="stExpander"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(139,92,246,0.15);
    }

    /* ── Section Headers ─────────────────────── */
    .section-hdr {
        display: flex; align-items: center; gap: 10px;
        margin: 25px 0 15px 0;
    }
    .section-hdr .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 10px rgba(139,92,246,0.5);
    }
    .section-hdr h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem; font-weight: 600;
        color: var(--text-primary); margin: 0;
    }

    /* ── Separator ────────────────────────────── */
    .glow-sep {
        height: 1px; margin: 30px 0;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.3), rgba(6,182,212,0.2), transparent);
    }

    /* ── Footer ───────────────────────────────── */
    .app-footer {
        text-align: center; padding: 35px 0 15px 0;
        margin-top: 50px;
        border-top: 1px solid var(--border);
    }
    .app-footer p {
        color: var(--text-muted); font-size: 0.78rem;
        letter-spacing: 0.5px;
    }
    .app-footer a {
        color: var(--accent); text-decoration: none;
    }

    /* ── Animations Stagger ──────────────────── */
    .delay-1 { animation-delay: 0.1s; }
    .delay-2 { animation-delay: 0.2s; }
    .delay-3 { animation-delay: 0.3s; }
    .delay-4 { animation-delay: 0.4s; }
</style>
""", unsafe_allow_html=True)


# ── Floating Background Orbs ─────────────────────────────────
st.markdown("""
<div class="orb-container">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL & ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("final_model.pkl")
    te_data = joblib.load("target_encoders.pkl")
    train_cols = joblib.load("train_columns.pkl")
    meta = joblib.load("model_metadata.pkl") if os.path.exists("model_metadata.pkl") else {}
    return model, te_data, train_cols, meta

try:
    model, te_data, train_cols, metadata = load_artifacts()
    target_encoders = te_data["encoders"]
    global_mean = te_data["global_mean"]
except FileNotFoundError as e:
    st.error(f"Missing artifact: {e}")
    st.stop()

model_name = metadata.get("model_name", "CatBoost")
r2_val = metadata.get("r2", 0)
rmse_val = metadata.get("rmse_rs", 0)

# Extract selectbox options dynamically from train_cols and target_encoders
def get_options(prefix, formatter=None):
    raw = [c[len(prefix)+1:] for c in train_cols if c.startswith(prefix + "_")]
    clean = []
    for r in raw:
        if r == "unknown":
            continue
        fmt = formatter(r) if formatter else r.title()
        clean.append(fmt)
    clean = sorted(list(set(clean)))
    if prefix != "category":
        clean = ["Unknown"] + clean
    return clean

category_options = get_options("category", lambda x: "Men's Watch" if "men" in x else "Women's Watch")
brand_options = get_options("brand")
watch_type_options = get_options("watch_type")
style_options = get_options("style")
belt_type_options = get_options("belt_type", lambda x: x.replace("_", " ").title())
case_material_options = get_options("case_material")
strap_material_options = get_options("strap_material")

# For target encoded columns, extract from target_encoders
dial_color_options = sorted([c.title() for c in target_encoders["dial_color"].index if c != "unknown"])
dial_color_options = ["Unknown"] + dial_color_options

strap_color_options = sorted([c.title() for c in target_encoders["strap_color"].index if c != "unknown"])
strap_color_options = ["Unknown"] + strap_color_options


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(f"""
<div class="hero-section">
    <div class="brand-badge">⚡ Powered by CatBoost AI</div>
    <h1 class="hero-title">Chronos AI</h1>
    <p class="hero-sub">
        Intelligent watch price prediction — trained on 1,488 watches
        from the WatchVine marketplace
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Strip ────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card delay-1">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-value">{r2_val:.1%}</div>
        <div class="kpi-label">Model Accuracy</div>
    </div>
    <div class="kpi-card delay-2">
        <div class="kpi-icon">📊</div>
        <div class="kpi-value">1,488</div>
        <div class="kpi-label">Watches Trained</div>
    </div>
    <div class="kpi-card delay-3">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-value">{len(train_cols)}</div>
        <div class="kpi-label">AI Features</div>
    </div>
    <div class="kpi-card delay-4">
        <div class="kpi-icon">💰</div>
        <div class="kpi-value">±₹{rmse_val:.0f}</div>
        <div class="kpi-label">Precision</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="glow-sep"></div>', unsafe_allow_html=True)


# ============================================================
# TABS — Predict | Analytics | About
# ============================================================

tab_predict, tab_analytics, tab_about = st.tabs(["⌚  Predict", "📊  Analytics", "ℹ️  About"])


# ============================================================
# TAB 1 — PREDICT
# ============================================================

with tab_predict:

    st.markdown("""
    <div class="section-hdr">
        <div class="dot"></div>
        <h3>Configure Watch Attributes</h3>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Form ───────────────────────────────────────────
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        category = st.selectbox("🏷️ Category", category_options)

        brand = st.selectbox("🏢 Brand", brand_options)

        watch_type = st.selectbox("⌚ Watch Type", watch_type_options)

        style = st.selectbox("🎨 Style", style_options)

        is_automatic = st.radio(
            "⚙️ Movement", ["Unknown", "Automatic", "Non-Automatic"],
            horizontal=True
        )

    with col_r:
        belt_type = st.selectbox("🔗 Belt Type", belt_type_options)

        case_material = st.selectbox("🛡️ Case Material", case_material_options)

        strap_material = st.selectbox("📿 Strap Material", strap_material_options)

        dial_color = st.selectbox("🎯 Dial Color", dial_color_options)

        strap_color = st.selectbox("🎨 Strap Color", strap_color_options)

    # ── Preprocessing ────────────────────────────────────────
    def preprocess(cat, brand, wtype, style, auto, belt, case_mat,
                   strap_mat, dial_c, strap_c):
        cl = lambda s: s.lower()
        cat_l = cl(cat)
        brand_l = cl(brand)
        wtype_l = cl(wtype)
        style_l = cl(style)
        belt_l = cl(belt).replace(" ", "_")
        case_l = cl(case_mat)
        strap_m_l = cl(strap_mat)
        dial_l = cl(dial_c)
        strap_c_l = cl(strap_c)

        # Map styles not in training set
        if style_l == "digital":
            style_l = "unknown"

        # Auto-derive redundant fields
        gender = "men" if cat_l == "men's watch" else (
            "women" if cat_l == "women's watch" else "unknown")
        watch_style_cat = wtype_l
        color = dial_l
        material = strap_m_l

        # Encode is_automatic
        auto_map = {"automatic": 1, "non-automatic": 0, "unknown": -1}
        auto_val = auto_map.get(cl(auto), -1)

        # Flags
        has_brand = int(brand_l != "unknown")
        has_details = int(color != "unknown")
        has_watch_type = int(wtype_l != "unknown")

        # Target encode
        def te(col, val):
            enc = target_encoders.get(col, {})
            r = enc.get(val, global_mean) if hasattr(enc, 'get') else global_mean
            return float(r) if not hasattr(r, '__iter__') or isinstance(r, str) else global_mean

        row = {c: 0 for c in train_cols}
        row["color"] = te("color", color)
        row["dial_color"] = te("dial_color", dial_l)
        row["strap_color"] = te("strap_color", strap_c_l)
        row["is_automatic"] = auto_val
        row["has_brand"] = has_brand
        row["has_details"] = has_details
        row["has_watch_type"] = has_watch_type

        for prefix, val in {
            "category": cat_l, "gender": gender, "belt_type": belt_l,
            "watch_type": wtype_l, "watch_style_category": watch_style_cat,
            "style": style_l, "case_material": case_l,
            "material": material, "strap_material": strap_m_l,
            "brand": brand_l,
        }.items():
            ohe = f"{prefix}_{val}"
            if ohe in row:
                row[ohe] = 1

        return pd.DataFrame([row])[train_cols]

    # ── Predict Button ───────────────────────────────────────
    st.markdown('<div class="glow-sep"></div>', unsafe_allow_html=True)

    if st.button("⚡ Generate Price Prediction", use_container_width=True, type="primary"):

        input_df = preprocess(
            category, brand, watch_type, style, is_automatic,
            belt_type, case_material, strap_material, dial_color, strap_color
        )

        log_pred = model.predict(input_df)[0]
        price = np.expm1(log_pred)
        p_low = max(price - rmse_val, 999)
        p_high = price + rmse_val

        # ── Result Card ──────────────────────────────────────
        st.markdown(f"""
        <div class="result-container">
            <div class="result-label">AI Price Estimate</div>
            <div class="result-price">₹{price:,.0f}</div>
            <div class="result-range">
                Confidence Range: <span>₹{p_low:,.0f}</span> — <span>₹{p_high:,.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics Row ──────────────────────────────────────
        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-pill">
                <div class="val">₹{price:,.0f}</div>
                <div class="lbl">Point Estimate</div>
            </div>
            <div class="metric-pill">
                <div class="val">₹{p_low:,.0f} — ₹{p_high:,.0f}</div>
                <div class="lbl">Confidence Range</div>
            </div>
            <div class="metric-pill">
                <div class="val">{log_pred:.4f}</div>
                <div class="lbl">Log Price</div>
            </div>
            <div class="metric-pill">
                <div class="val">{r2_val:.1%}</div>
                <div class="lbl">Model R²</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── AI Insights ──────────────────────────────────────
        insights = []
        if brand.lower() != "unknown":
            insights.append(("🏢", f"<strong>{brand}</strong> brand detected — brand identity significantly influences pricing."))
        if watch_type.lower() == "luxury":
            insights.append(("💎", "Luxury watch type selected — expect <strong>premium pricing</strong> in the higher range."))
        if case_material.lower() in ["gold", "rose gold", "titanium"]:
            insights.append(("🛡️", f"<strong>{case_material}</strong> case material is a premium indicator — pushes price upward."))
        if is_automatic.lower() == "automatic":
            insights.append(("⚙️", "Automatic movement is a <strong>premium feature</strong> — typical of higher-end watches."))
        if not insights:
            insights.append(("🔍", "Most attributes are set to <strong>Unknown</strong> — add more details for a more precise prediction."))

        st.markdown("""
        <div class="section-hdr">
            <div class="dot"></div>
            <h3>AI Insights</h3>
        </div>
        """, unsafe_allow_html=True)

        for icon, text in insights:
            st.markdown(f"""
            <div class="insight-card">
                <div class="icon">{icon}</div>
                <div class="text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Confidence Gauge ─────────────────────────────────
        st.markdown("""
        <div class="section-hdr">
            <div class="dot"></div>
            <h3>Prediction Confidence</h3>
        </div>
        """, unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=r2_val * 100,
            number={"suffix": "%", "font": {"size": 42, "color": "#c4b5fd", "family": "Space Grotesk"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.15)",
                         "tickfont": {"color": "rgba(255,255,255,0.3)"}},
                "bar": {"color": "#8b5cf6", "thickness": 0.3},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(239,68,68,0.1)"},
                    {"range": [30, 60], "color": "rgba(245,158,11,0.1)"},
                    {"range": [60, 100], "color": "rgba(34,197,94,0.1)"},
                ],
                "threshold": {
                    "line": {"color": "#06b6d4", "width": 3},
                    "thickness": 0.8, "value": r2_val * 100,
                },
            },
        ))
        gauge.update_layout(
            height=280,
            margin=dict(l=30, r=30, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "rgba(255,255,255,0.5)"},
        )
        st.plotly_chart(gauge, use_container_width=True)

        # ── SHAP Explanation ─────────────────────────────────
        st.markdown("""
        <div class="section-hdr">
            <div class="dot"></div>
            <h3>Feature Impact Analysis (SHAP)</h3>
        </div>
        """, unsafe_allow_html=True)

        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            shap_df = pd.DataFrame({
                "Feature": train_cols,
                "SHAP": shap_values[0]
            })
            shap_df["Abs"] = shap_df["SHAP"].abs()
            top = shap_df.nlargest(15, "Abs").sort_values("SHAP")

            colors = ["#ef4444" if v < 0 else "#8b5cf6" for v in top["SHAP"]]

            fig = go.Figure(go.Bar(
                x=top["SHAP"].values,
                y=top["Feature"].values,
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(width=0),
                    cornerradius=4,
                ),
                hovertemplate="<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>"
            ))
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=20, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="rgba(255,255,255,0.6)", size=11),
                xaxis=dict(
                    gridcolor="rgba(255,255,255,0.04)",
                    zerolinecolor="rgba(255,255,255,0.1)",
                    title="SHAP Value (impact on price)",
                ),
                yaxis=dict(gridcolor="rgba(255,255,255,0.02)"),
                hoverlabel=dict(
                    bgcolor="#1a1a3e",
                    font_size=12,
                    bordercolor="rgba(139,92,246,0.3)",
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="insight-card">
                <div class="icon">💡</div>
                <div class="text">
                    <strong>Purple bars</strong> push the price UP.
                    <strong style="color:#ef4444;">Red bars</strong> push the price DOWN.
                    The longer the bar, the stronger the impact on the prediction.
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception:
            st.info("SHAP analysis requires the model to support TreeExplainer.")

        # ── Input Summary ────────────────────────────────────
        with st.expander("📋 View Full Input Configuration"):
            summary_data = {
                "Attribute": ["Category", "Brand", "Watch Type", "Style", "Movement",
                              "Belt Type", "Case Material", "Strap Material",
                              "Dial Color", "Strap Color"],
                "Selected Value": [category, brand, watch_type, style, is_automatic,
                                   belt_type, case_material, strap_material,
                                   dial_color, strap_color],
            }
            st.dataframe(
                pd.DataFrame(summary_data),
                use_container_width=True, hide_index=True
            )

    else:
        # ── Empty State ──────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding:60px 0 40px 0;">
            <div style="font-size:4rem; margin-bottom:15px; opacity:0.3;">⌚</div>
            <p style="color:var(--text-secondary); font-size:1.1rem; font-weight:400;">
                Configure watch attributes above
            </p>
            <p style="color:var(--text-muted); font-size:0.88rem; margin-top:5px;">
                Click <strong style="color:#a78bfa;">Generate Price Prediction</strong> to get an AI-powered estimate
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 2 — ANALYTICS
# ============================================================

with tab_analytics:

    st.markdown("""
    <div class="section-hdr">
        <div class="dot"></div>
        <h3>Model Performance Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)

    # ── Performance KPIs ─────────────────────────────────────
    rmse_log = metadata.get("rmse_log", 0.178)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-icon">🤖</div>
            <div class="kpi-value">{model_name}</div>
            <div class="kpi-label">Final Model</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-value">{r2_val:.4f}</div>
            <div class="kpi-label">R² Score</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">📏</div>
            <div class="kpi-value">₹{rmse_val:.0f}</div>
            <div class="kpi-label">RMSE (₹)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">📐</div>
            <div class="kpi-value">{rmse_log:.4f}</div>
            <div class="kpi-label">RMSE (Log)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Comparison Chart ───────────────────────────────
    st.markdown("""
    <div class="section-hdr">
        <div class="dot"></div>
        <h3>Model Comparison (6 Models Evaluated)</h3>
    </div>
    """, unsafe_allow_html=True)

    models_data = {
        "CatBoost": 0.4544, "Random Forest": 0.4273,
        "XGBoost": 0.4209, "Ridge": 0.4121,
        "Linear Reg.": 0.4120, "LightGBM": 0.3941,
    }

    m_names = list(models_data.keys())[::-1]
    m_scores = list(models_data.values())[::-1]
    m_colors = ["#8b5cf6" if n == "CatBoost" else "rgba(139,92,246,0.25)" for n in m_names]

    fig_comp = go.Figure(go.Bar(
        x=m_scores, y=m_names, orientation="h",
        marker=dict(color=m_colors, cornerradius=6,
                    line=dict(width=0)),
        text=[f"{s:.4f}" for s in m_scores],
        textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.6)", size=12,
                      family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>R² = %{x:.4f}<extra></extra>",
    ))
    fig_comp.update_layout(
        height=320,
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.6)", size=12),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            range=[0, 0.55],
            title="R² Score (higher is better)",
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0.02)"),
        hoverlabel=dict(bgcolor="#1a1a3e", bordercolor="rgba(139,92,246,0.3)"),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Pipeline Overview ────────────────────────────────────
    st.markdown("""
    <div class="section-hdr">
        <div class="dot"></div>
        <h3>ML Pipeline Overview</h3>
    </div>
    """, unsafe_allow_html=True)

    pipe_cols = st.columns(4)
    pipeline_steps = [
        ("🧹", "Data Cleaning", "3,000+ → 18 cols"),
        ("🔍", "EDA", "25 plots, insights"),
        ("⚙️", "Feature Eng.", "96 features built"),
        ("🤖", "Model + Tuning", "CatBoost, Optuna"),
    ]
    for col, (icon, title, desc) in zip(pipe_cols, pipeline_steps):
        col.markdown(f"""
        <div class="g-card" style="text-align:center; min-height:140px;">
            <div style="font-size:2rem; margin-bottom:10px;">{icon}</div>
            <div style="color:var(--text-primary); font-weight:600; font-size:0.9rem;">{title}</div>
            <div style="color:var(--text-muted); font-size:0.78rem; margin-top:6px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Data Quality Note ────────────────────────────────────
    st.markdown("""
    <div class="insight-card" style="margin-top:20px;">
        <div class="icon">⚠️</div>
        <div class="text">
            <strong>Data quality challenge:</strong> 87.5% of watches have unknown brand and
            57.3% lack attribute details. The model treats 'unknown' as a valid category,
            which limits R² but prevents data fabrication.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 3 — ABOUT
# ============================================================

with tab_about:

    st.markdown("""
    <div class="section-hdr">
        <div class="dot"></div>
        <h3>About Chronos AI</h3>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2, gap="large")

    with a1:
        st.markdown("""
        <div class="g-card">
            <div class="card-label">Project</div>
            <p style="color:var(--text-primary); font-size:1.1rem; font-weight:600; margin:5px 0;">
                Watch Price Prediction
            </p>
            <p style="color:var(--text-secondary); font-size:0.88rem; line-height:1.7;">
                An end-to-end machine learning pipeline that predicts watch prices from
                the WatchVine Indian e-commerce dataset. Built as a Summer Internship
                project with a focus on production-quality code, leakage-free engineering,
                and deployment.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="g-card">
            <div class="card-label">Author</div>
            <p style="color:var(--text-primary); font-weight:600; margin:5px 0;">
                Rudra
            </p>
            <p style="color:var(--text-secondary); font-size:0.88rem;">
                GitHub: <a href="https://github.com/Rudra2986" target="_blank"
                style="color:#a78bfa;">@Rudra2986</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="g-card">
            <div class="card-label">Tech Stack</div>
            <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:10px;">
        """ + "".join([
            f'<span style="background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.2); border-radius:8px; padding:5px 12px; font-size:0.78rem; color:#c4b5fd; font-weight:500;">{t}</span>'
            for t in ["Python 3.11", "CatBoost", "Scikit-Learn", "XGBoost",
                       "LightGBM", "Optuna", "SHAP", "Streamlit", "Plotly",
                       "Pandas", "NumPy", "Matplotlib"]
        ]) + """
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="g-card">
            <div class="card-label">Key Design Decisions</div>
            <ul style="color:var(--text-secondary); font-size:0.85rem; line-height:1.8; padding-left:18px;">
                <li>'Unknown' kept as valid category — not imputed</li>
                <li>Train/test split <strong>before</strong> encoding (leakage-free)</li>
                <li>Log-transformed target for better convergence</li>
                <li>Bayesian optimization via Optuna (100 trials)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="app-footer">
    <p>
        Chronos AI · Watch Price Intelligence ·
        <a href="https://github.com/Rudra2986/watch-price-prediction" target="_blank">GitHub</a>
        · Built by Rudra · Summer Internship 2026
    </p>
</div>
""", unsafe_allow_html=True)
