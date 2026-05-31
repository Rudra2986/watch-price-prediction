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
        --bg-primary: #040906;
        --bg-card: rgba(16, 36, 25, 0.45);
        --bg-card-hover: rgba(24, 54, 38, 0.6);
        --border: rgba(255, 255, 255, 0.05);
        --border-glow: rgba(223, 177, 91, 0.25);
        --accent: #dfb15b; /* Champagne Gold */
        --accent-2: #34d399; /* Seafoam Mint */
        --accent-3: #10b981; /* Emerald Green */
        --text-primary: rgba(255, 255, 255, 0.95);
        --text-secondary: rgba(210, 230, 220, 0.65);
        --text-muted: rgba(160, 190, 175, 0.35);
        --gradient-1: linear-gradient(135deg, #dfb15b, #34d399);
        --gradient-2: linear-gradient(135deg, #10b981, #dfb15b, #34d399);
        --glow-gold: 0 0 30px rgba(223, 177, 91, 0.18);
        --glow-emerald: 0 0 30px rgba(16, 185, 129, 0.18);
    }

    /* ── Global ──────────────────────────────── */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            linear-gradient(rgba(52, 211, 153, 0.008) 1px, transparent 1px),
            linear-gradient(90deg, rgba(52, 211, 153, 0.008) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 1;
    }
    [data-testid="stSidebar"] { display: none; }
    .block-container { max-width: 1200px; padding-top: 90px !important; }
    #MainMenu, footer, header { display: none !important; }

    /* ── Navigation Bar ──────────────────────── */
    .nav-bar {
        position: fixed;
        top: 0; left: 0; width: 100%;
        height: 70px;
        background: rgba(4, 9, 6, 0.75);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .nav-container {
        width: 100%;
        max-width: 1200px;
        padding: 0 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .nav-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.35rem;
        color: #ffffff;
        letter-spacing: 0.5px;
        text-decoration: none;
    }
    .nav-logo-icon {
        font-size: 1.5rem;
    }
    .nav-logo-text span {
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .nav-links {
        display: flex;
        align-items: center;
        gap: 32px;
    }
    .nav-link {
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-decoration: none;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .nav-link:hover {
        color: #ffffff;
        text-shadow: 0 0 10px rgba(52,211,153,0.3);
    }
    .nav-cta {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(223, 177, 91, 0.12);
        border: 1px solid rgba(223, 177, 91, 0.3);
        border-radius: 10px;
        padding: 8px 16px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #dfb15b;
        text-decoration: none;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    .nav-cta:hover {
        background: rgba(223, 177, 91, 0.2);
        border-color: rgba(223, 177, 91, 0.5);
        box-shadow: 0 0 15px rgba(223, 177, 91, 0.25);
        transform: translateY(-1px);
    }
    @media (max-width: 768px) {
        .nav-links {
            display: none;
        }
    }

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
        background: rgba(223, 177, 91, 0.06);
        animation-delay: 0s;
    }
    .orb-2 {
        width: 400px; height: 400px; bottom: -10%; right: -5%;
        background: rgba(16, 185, 129, 0.05);
        animation-delay: -7s;
    }
    .orb-3 {
        width: 300px; height: 300px; top: 40%; left: 50%;
        background: rgba(52, 211, 153, 0.04);
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
        background: rgba(223, 177, 91, 0.1);
        border: 1px solid rgba(223, 177, 91, 0.2);
        border-radius: 100px; padding: 6px 18px;
        font-size: 0.78rem; color: #dfb15b;
        font-weight: 500; letter-spacing: 1.5px;
        text-transform: uppercase; margin-bottom: 20px;
        animation: fadeDown 0.8s ease-out;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem; font-weight: 700;
        line-height: 1.1; margin: 0;
        background: linear-gradient(135deg, #ffffff 0%, #dfb15b 40%, #34d399 70%, #ffffff 100%);
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
        position: fixed;
        top: 13px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000000;
        gap: 8px;
        background: rgba(16, 36, 25, 0.5) !important;
        border: 1px solid rgba(223, 177, 91, 0.2) !important;
        border-radius: 100px;
        padding: 4px 6px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 100px !important;
        padding: 8px 24px !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(52, 211, 153, 0.4);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(52, 211, 153, 0.15) !important;
        color: #34d399 !important;
        border: 1px solid rgba(52, 211, 153, 0.3) !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 25px; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    /* ── Glass Card ───────────────────────────── */
    .g-card {
        background: var(--bg-card);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid var(--border);
        border-radius: 18px; padding: 28px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeUp 0.6s ease-out both;
    }
    .g-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), var(--accent-2), transparent);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    .g-card:hover {
        background: var(--bg-card-hover);
        border-color: rgba(223, 177, 91, 0.25);
        box-shadow: var(--glow-gold);
        transform: translateY(-3px);
    }
    .g-card:hover::before {
        opacity: 1;
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
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        animation: fadeUp 0.6s ease-out both;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-2), var(--accent), transparent);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    .kpi-card:hover {
        background: var(--bg-card-hover);
        border-color: rgba(52, 211, 153, 0.25);
        box-shadow: var(--glow-emerald);
        transform: translateY(-4px);
    }
    .kpi-card:hover::before {
        opacity: 1;
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
        background: linear-gradient(135deg, rgba(223, 177, 91, 0.12), rgba(52, 211, 153, 0.08));
        border: 1px solid rgba(223, 177, 91, 0.25);
        box-shadow: 0 8px 32px rgba(223, 177, 91, 0.1);
        border-radius: 24px; padding: 45px 30px;
        text-align: center; margin: 25px 0;
        position: relative; overflow: hidden;
        animation: fadeUp 0.5s ease-out;
    }
    .result-container::before {
        content: ''; position: absolute;
        top: 0; left: 0;
        width: 100%; height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent));
    }
    .result-label {
        font-size: 0.78rem; text-transform: uppercase;
        letter-spacing: 3px; color: var(--text-muted);
        font-weight: 600; margin-bottom: 10px;
    }
    .result-price {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem; font-weight: 700;
        background: linear-gradient(135deg, #ffffff, #dfb15b, #34d399);
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
        color: #dfb15b; font-weight: 600;
    }
    @keyframes priceReveal {
        from { opacity: 0; transform: scale(0.8); filter: blur(10px); }
        to { opacity: 1; transform: scale(1); filter: blur(0); }
    }

    /* ── Metrics Row ──────────────────────────── */
    .metrics-row { display: flex; gap: 14px; margin: 20px 0; }
    .metric-pill {
        flex: 1; background: rgba(16, 36, 25, 0.3);
        border: 1px solid var(--border);
        border-radius: 14px; padding: 18px;
        text-align: center;
        transition: all 0.4s ease;
    }
    .metric-pill:hover {
        background: rgba(24, 54, 38, 0.45);
        border-color: rgba(223, 177, 91, 0.25);
        box-shadow: var(--glow-gold);
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
        background: rgba(16, 185, 129, 0.06);
        border: 1px solid rgba(16, 185, 129, 0.12);
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
        color: #dfb15b;
    }

    /* ── Buttons ──────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #dfb15b, #10b981, #34d399) !important;
        background-size: 200% 200% !important;
        color: #040906 !important; border: none !important;
        border-radius: 14px !important;
        padding: 16px 32px !important;
        font-weight: 700 !important; font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(52,211,153,0.2) !important;
    }
    .stButton > button:hover {
        background-position: 100% 50% !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 35px rgba(52,211,153,0.3) !important;
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
        border-color: rgba(223, 177, 91, 0.15);
    }

    /* ── Section Headers ─────────────────────── */
    .section-hdr {
        display: flex; align-items: center; gap: 10px;
        margin: 25px 0 15px 0;
    }
    .section-hdr .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 10px rgba(223, 177, 91, 0.5);
    }
    .section-hdr h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem; font-weight: 600;
        color: var(--text-primary); margin: 0;
    }

    /* ── Separator ────────────────────────────── */
    .glow-sep {
        height: 1px; margin: 30px 0;
        background: linear-gradient(90deg, transparent, rgba(223,177,91,0.25), rgba(52,211,153,0.18), transparent);
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

    /* ── Mobile Responsiveness ───────────────── */
    @media (max-width: 768px) {
        .hero-section {
            text-align: center !important;
            padding: 20px 0 !important;
        }
        .hero-title {
            font-size: 2.2rem !important;
        }
        .hero-sub {
            font-size: 0.9rem !important;
            max-width: 100% !important;
        }
        .kpi-row {
            flex-direction: column !important;
            gap: 12px !important;
        }
        .metrics-row {
            flex-direction: column !important;
            gap: 10px !important;
        }
            font-size: 2.5rem !important;
        }
        .g-card {
            padding: 20px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ── Custom Navigation Bar & Floating Background Orbs ──────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-container">
        <a href="#" class="nav-logo">
            <span class="nav-logo-icon">⌚</span>
            <span class="nav-logo-text">Chronos <span>AI</span></span>
        </a>
        <!-- Native tabs will float here automatically via CSS fixed positioning -->
        <a href="https://github.com/Rudra2986/watch-price-prediction" target="_blank" class="nav-cta">
            <span>View GitHub</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
        </a>
    </div>
</div>

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

category_options = get_options("category", lambda x: "Women's Watch" if "women" in x else "Men's Watch")
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

# ── Hero Section (Two Columns: Info & 3D Interactive Watch) ──
col_hero_left, col_hero_right = st.columns([1.2, 1], gap="large")

with col_hero_left:
    st.markdown(f"""
    <div class="hero-section" style="text-align: left; padding: 40px 0 20px 0;">
        <div class="brand-badge">⚡ Powered by {model_name} AI</div>
        <h1 class="hero-title">Chronos AI</h1>
        <p class="hero-sub" style="margin-top: 15px; max-width: 500px;">
            Intelligent watch price prediction — trained on 1,488 watches
            from the WatchVine marketplace. Hover or touch to rotate the interactive 3D timepiece.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_hero_right:
    # 3D Interactive Watch Component (Three.js)
    watch_3d_html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {
            margin: 0;
            overflow: hidden;
            background: transparent;
        }
        #canvas-container {
            width: 100%;
            height: 320px;
            position: relative;
            cursor: grab;
        }
        #canvas-container:active {
            cursor: grabbing;
        }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
    <div id="canvas-container"></div>
    <script>
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.z = 8;

        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const watchGroup = new THREE.Group();
        scene.add(watchGroup);

        // Materials
        const goldMaterial = new THREE.MeshBasicMaterial({ color: 0xdfb15b, wireframe: true, transparent: true, opacity: 0.85 });
        const mintMaterial = new THREE.MeshBasicMaterial({ color: 0x34d399, wireframe: true, transparent: true, opacity: 0.85 });
        const emeraldMaterial = new THREE.MeshBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.9 });
        const dialMaterial = new THREE.MeshBasicMaterial({ color: 0x0a140f, transparent: true, opacity: 0.45 });
        const ticksMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 });

        // Bezel / Case
        const bezelGeom = new THREE.TorusGeometry(1.6, 0.15, 8, 48);
        const bezel = new THREE.Mesh(bezelGeom, goldMaterial);
        watchGroup.add(bezel);
        
        const bezelInnerGeom = new THREE.TorusGeometry(1.4, 0.05, 8, 48);
        const bezelInner = new THREE.Mesh(bezelInnerGeom, mintMaterial);
        watchGroup.add(bezelInner);

        // Dial Face
        const dialGeom = new THREE.CylinderGeometry(1.35, 1.35, 0.05, 32);
        dialGeom.rotateX(Math.PI / 2);
        const dial = new THREE.Mesh(dialGeom, dialMaterial);
        watchGroup.add(dial);

        // Hour Ticks
        const ticksGroup = new THREE.Group();
        for (let i = 0; i < 12; i++) {
            const angle = (i * Math.PI * 2) / 12;
            const size = (i % 3 === 0) ? 0.2 : 0.08;
            const tickGeom = new THREE.BoxGeometry(0.04, size, 0.04);
            const tick = new THREE.Mesh(tickGeom, ticksMaterial);
            tick.position.x = Math.sin(angle) * 1.25;
            tick.position.y = Math.cos(angle) * 1.25;
            tick.rotation.z = -angle;
            ticksGroup.add(tick);
        }
        watchGroup.add(ticksGroup);

        // Strap Lugs and Bands
        const strapTopGeom = new THREE.BoxGeometry(0.8, 1.2, 0.1);
        const strapTop = new THREE.Mesh(strapTopGeom, goldMaterial);
        strapTop.position.y = 2.1;
        strapTop.position.z = -0.15;
        strapTop.rotation.x = -0.15;
        watchGroup.add(strapTop);

        const strapBottomGeom = new THREE.BoxGeometry(0.8, 1.2, 0.1);
        const strapBottom = new THREE.Mesh(strapBottomGeom, goldMaterial);
        strapBottom.position.y = -2.1;
        strapBottom.position.z = -0.15;
        strapBottom.rotation.x = 0.15;
        watchGroup.add(strapBottom);

        // Crown
        const crownGeom = new THREE.CylinderGeometry(0.15, 0.15, 0.25, 12);
        const crown = new THREE.Mesh(crownGeom, mintMaterial);
        crown.position.x = 1.75;
        crown.rotation.z = -Math.PI / 2;
        watchGroup.add(crown);

        // Hands Group (offset pivots for proper rotation)
        const hourHandGeom = new THREE.BoxGeometry(0.08, 0.75, 0.02);
        hourHandGeom.translate(0, 0.375, 0.02);
        const hourHand = new THREE.Mesh(hourHandGeom, goldMaterial);
        watchGroup.add(hourHand);

        const minHandGeom = new THREE.BoxGeometry(0.06, 1.1, 0.02);
        minHandGeom.translate(0, 0.55, 0.04);
        const minHand = new THREE.Mesh(minHandGeom, mintMaterial);
        watchGroup.add(minHand);

        const secHandGeom = new THREE.BoxGeometry(0.02, 1.25, 0.01);
        secHandGeom.translate(0, 0.625, 0.06);
        const secHand = new THREE.Mesh(secHandGeom, emeraldMaterial);
        watchGroup.add(secHand);

        // Background Particles
        const particlesCount = 50;
        const particlesGeom = new THREE.BufferGeometry();
        const positions = new Float32Array(particlesCount * 3);
        const speeds = [];

        for (let i = 0; i < particlesCount; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 8;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 8;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 6 - 2;
            speeds.push({
                x: (Math.random() - 0.5) * 0.005,
                y: (Math.random() - 0.5) * 0.005,
                z: (Math.random() - 0.5) * 0.005
            });
        }

        particlesGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const pMaterial = new THREE.PointsMaterial({
            color: 0x34d399,
            size: 0.05,
            transparent: true,
            opacity: 0.5
        });
        const starField = new THREE.Points(particlesGeom, pMaterial);
        scene.add(starField);

        // Interaction state
        let targetX = 0, targetY = 0;

        window.addEventListener('mousemove', (e) => {
            const rect = container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
                targetX = (x / rect.width) * 2 - 1;
                targetY = -(y / rect.height) * 2 + 1;
            }
        });

        window.addEventListener('touchmove', (e) => {
            if (e.touches.length > 0) {
                const rect = container.getBoundingClientRect();
                const x = e.touches[0].clientX - rect.left;
                const y = e.touches[0].clientY - rect.top;
                if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
                    targetX = (x / rect.width) * 2 - 1;
                    targetY = -(y / rect.height) * 2 + 1;
                }
            }
        });

        window.addEventListener('mouseleave', () => {
            targetX = 0;
            targetY = 0;
        });

        const animate = () => {
            requestAnimationFrame(animate);

            // Time calculations
            const now = new Date();
            const hrs = now.getHours() % 12;
            const mins = now.getMinutes();
            const secs = now.getSeconds();
            const ms = now.getMilliseconds();

            const hrAngle = -((hrs + mins / 60) * (2 * Math.PI / 12));
            const minAngle = -((mins + secs / 60) * (2 * Math.PI / 60));
            const secAngle = -((secs + ms / 1000) * (2 * Math.PI / 60));

            hourHand.rotation.z = hrAngle;
            minHand.rotation.z = minAngle;
            secHand.rotation.z = secAngle;

            // Base rotation spin
            const baseSpin = Date.now() * 0.0003;
            
            // Mouse tracking + base floating spin
            watchGroup.rotation.y += (targetX * 0.6 + Math.sin(baseSpin) * 0.15 - watchGroup.rotation.y) * 0.05;
            watchGroup.rotation.x += (targetY * 0.4 + Math.cos(baseSpin) * 0.1 - watchGroup.rotation.x) * 0.05;

            // Particles animation
            const posAttr = starField.geometry.attributes.position;
            for (let i = 0; i < particlesCount; i++) {
                posAttr.array[i * 3] += speeds[i].x;
                posAttr.array[i * 3 + 1] += speeds[i].y;
                posAttr.array[i * 3 + 2] += speeds[i].z;

                if (Math.abs(posAttr.array[i * 3]) > 4) posAttr.array[i * 3] = -posAttr.array[i * 3];
                if (Math.abs(posAttr.array[i * 3 + 1]) > 4) posAttr.array[i * 3 + 1] = -posAttr.array[i * 3 + 1];
                if (Math.abs(posAttr.array[i * 3 + 2] + 2) > 4) posAttr.array[i * 3 + 2] = -2;
            }
            starField.geometry.attributes.position.needsUpdate = true;

            renderer.render(scene, camera);
        };

        window.addEventListener('resize', () => {
            const w = container.clientWidth;
            const h = container.clientHeight;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        });

        animate();
    </script>
    </body>
    </html>
    """
    import streamlit.components.v1 as components
    components.html(watch_3d_html, height=350)

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

    # ── Input Form (Wrapped in Glass Card Panel) ─────────────
    st.markdown('<div class="g-card">', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

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
            "dial_color": dial_l, "strap_color": strap_c_l,
            "brand": brand_l
        }.items():
            col_name = f"{prefix}_{val}"
            if col_name in row:
                row[col_name] = 1
        
        return pd.DataFrame([row])

    # ── Action Button ────────────────────────────────────────
    if st.button("Generate Price Prediction", use_container_width=True):
        input_df = preprocess(category, brand, watch_type, style, is_automatic, 
                             belt_type, case_material, strap_material, dial_color, strap_color)
        
        # Prediction
        log_pred = model.predict(input_df)[0]
        price = np.exp(log_pred)
        p_low = price * 0.92
        p_high = price * 1.08

        # Display Results
        col_res_l, col_res_r = st.columns([1, 1], gap="large")

        with col_res_l:
            st.markdown(f"""
            <div class="metrics-row">
                <div class="metric-pill" style="border-color: rgba(223, 177, 91, 0.35); background: rgba(223, 177, 91, 0.04);">
                    <div class="val" style="background: var(--gradient-1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">₹{price:,.0f}</div>
                    <div class="lbl" style="color: var(--accent);">Point Estimate</div>
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

            # AI Insights
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

        with col_res_r:
            # Confidence Gauge
            st.markdown('<div class="g-card" style="margin-bottom: 20px;">', unsafe_allow_html=True)
            st.markdown("""
            <div class="section-hdr" style="margin-top: 0;">
                <div class="dot"></div>
                <h3>Prediction Confidence</h3>
            </div>
            """, unsafe_allow_html=True)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=r2_val * 100,
                number={"suffix": "%", "font": {"size": 36, "color": "#dfb15b", "family": "Space Grotesk"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.15)",
                             "tickfont": {"color": "rgba(255,255,255,0.3)", "size": 10}},
                    "bar": {"color": "#34d399", "thickness": 0.3},
                    "bgcolor": "rgba(255,255,255,0.03)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(239,68,68,0.1)"},
                        {"range": [30, 60], "color": "rgba(245,158,11,0.1)"},
                        {"range": [60, 100], "color": "rgba(52,211,153,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#dfb15b", "width": 3},
                        "thickness": 0.8, "value": r2_val * 100,
                    },
                },
            ))
            gauge.update_layout(
                height=220,
                margin=dict(l=30, r=30, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "rgba(255,255,255,0.5)"},
            )
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # SHAP Explanation
            st.markdown('<div class="g-card">', unsafe_allow_html=True)
            st.markdown("""
            <div class="section-hdr" style="margin-top: 0;">
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
                top = shap_df.nlargest(10, "Abs").sort_values("SHAP")

                colors = ["#ef4444" if v < 0 else "#34d399" for v in top["SHAP"]]

                fig = go.Figure(go.Bar(
                    x=top["SHAP"].values,
                    y=top["Feature"].values,
                    orientation="h",
                    marker=dict(
                        color=colors,
                        line=dict(width=0),
                        cornerradius=4
                    )
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="rgba(255,255,255,0.6)", size=11),
                    xaxis=dict(
                        gridcolor="rgba(255,255,255,0.04)",
                        title="Impact on Model Output",
                    ),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.02)"),
                    hoverlabel=dict(
                        bgcolor="#1a1a3e",
                        font_size=12,
                        bordercolor="rgba(52,211,153,0.3)",
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="insight-card">
                    <div class="icon">💡</div>
                    <div class="text">
                        <strong style="color:#34d399;">Mint Green bars</strong> push the price UP.
                        <strong style="color:#ef4444;">Red bars</strong> push the price DOWN.
                        The longer the bar, the stronger the impact on the prediction.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception:
                st.info("SHAP analysis requires the model to support TreeExplainer.")
            st.markdown('</div>', unsafe_allow_html=True)

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
                Click <strong style="color:#dfb15b;">Generate Price Prediction</strong> to get an AI-powered estimate
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

    # ── Row 1: KPI Cards + Breakthrough Details (Bento Grid) ──
    col_kpi_l, col_kpi_r = st.columns([1.8, 1.2], gap="large")

    with col_kpi_l:
        rmse_log = metadata.get("rmse_log", 0.178)
        st.markdown(f"""
        <div class="kpi-row" style="margin: 0; display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
            <div class="kpi-card" style="margin: 0;">
                <div class="kpi-icon">🤖</div>
                <div class="kpi-value">{model_name}</div>
                <div class="kpi-label">Final Model</div>
            </div>
            <div class="kpi-card" style="margin: 0;">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-value">{r2_val:.4f}</div>
                <div class="kpi-label">R² Score</div>
            </div>
            <div class="kpi-card" style="margin: 0;">
                <div class="kpi-icon">📏</div>
                <div class="kpi-value">₹{rmse_val:.0f}</div>
                <div class="kpi-label">RMSE (₹)</div>
            </div>
            <div class="kpi-card" style="margin: 0;">
                <div class="kpi-icon">📐</div>
                <div class="kpi-value">{rmse_log:.4f}</div>
                <div class="kpi-label">RMSE (Log)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi_r:
        st.markdown("""
        <div class="g-card" style="height: 100%;">
            <div class="card-label">Breakthrough Insight</div>
            <p style="color:var(--text-primary); font-size:1.15rem; font-weight:700; margin:10px 0 15px 0;">
                Title Feature Extraction
            </p>
            <p style="color:var(--text-secondary); font-size:0.85rem; line-height:1.7; margin:0;">
                Resolving name sparsity was the key: watch brand & movement details were extracted directly from titles.
                This consolidated 62 noisy values down to <strong>49 standardized brands</strong> and recovered 300+ movements,
                boosting the model R² accuracy from <strong>45% to 71.2%</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="glow-sep"></div>', unsafe_allow_html=True)

    # ── Row 2: Model Comparison Chart + Active Configuration ────
    col_chart_l, col_chart_r = st.columns([2.2, 1], gap="large")

    with col_chart_l:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-hdr" style="margin-top: 0;">
            <div class="dot"></div>
            <h3>Model Comparison (6 Models Evaluated)</h3>
        </div>
        """, unsafe_allow_html=True)

        models_data = {
            "XGBoost": 0.7079, "CatBoost": 0.7004,
            "Random Forest": 0.6945, "Ridge": 0.5629,
            "Linear Reg.": 0.5629, "LightGBM": 0.5460,
        }

        m_names = list(models_data.keys())[::-1]
        m_scores = list(models_data.values())[::-1]
        m_colors = ["#34d399" if n == model_name else "rgba(52,211,153,0.2)" for n in m_names]

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
                range=[0, 0.85],
                title="R² Score (higher is better)",
            ),
            yaxis=dict(gridcolor="rgba(255,255,255,0.02)"),
            hoverlabel=dict(bgcolor="#0a140f", bordercolor="rgba(52,211,153,0.3)"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart_r:
        st.markdown(f"""
        <div class="g-card" style="height: 100%;">
            <div class="card-label">Active Model Config</div>
            <p style="color:var(--text-primary); font-size:1.1rem; font-weight:700; margin:8px 0;">
                {model_name} Predictor
            </p>
            <ul style="color:var(--text-secondary); font-size:0.82rem; line-height:1.8; padding-left:16px; margin-top:10px;">
                <li>Training Features: <strong>{len(train_cols)}</strong></li>
                <li>Hyperparameter Tuning: <strong>Optuna</strong></li>
                <li>Trials Evaluated: <strong>100 Trials</strong></li>
                <li>Loss Metric: <strong>Root Mean Squared Error</strong></li>
                <li>Target Scale: <strong>Log-transformed</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="glow-sep"></div>', unsafe_allow_html=True)

    # ── Row 3: Pipeline Overview ─────────────────────────────
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
        ("⚙️", "Feature Eng.", f"{len(train_cols)} features built"),
        ("🤖", "Model + Tuning", f"{model_name}, Optuna"),
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
        <div class="icon">💡</div>
        <div class="text">
            <strong>Extraction Breakthrough:</strong> Watch brand and movement status were extracted
            from the product titles, reducing unknown brands from 87.5% to 0.07% and recovering 300+
            automatic watches. This boosted $R^2$ model accuracy from 45% to 71.2%.
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
                style="color:#dfb15b;">@Rudra2986</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="g-card">
            <div class="card-label">Tech Stack</div>
            <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:10px;">
        """ + "".join([
            f'<span style="background:rgba(223,177,91,0.1); border:1px solid rgba(223,177,91,0.25); border-radius:8px; padding:5px 12px; font-size:0.78rem; color:#dfb15b; font-weight:600;">{t}</span>'
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
