# ============================================================
# STEP 7 — STREAMLIT DEPLOYMENT (v4 — Professional SaaS UI)
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

# ── Theme State Management ──────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Floating toggle switch in the top-right navbar area (styled with fixed position in CSS)
theme_toggle = st.toggle("Dark Theme", value=(st.session_state.theme == "dark"), label_visibility="collapsed")
st.session_state.theme = "dark" if theme_toggle else "light"

# Define Theme CSS Variables
if st.session_state.theme == "dark":
    theme_variables = """
    :root {
        --bg-primary: #0A140F;
        --bg-card: #12221A;
        --bg-card-hover: #162B20;
        --bg-warm: #101F18;
        --border: rgba(201, 169, 110, 0.2);
        --border-hover: rgba(201, 169, 110, 0.45);
        --accent: #C9A96E;
        --accent-light: rgba(201, 169, 110, 0.12);
        --accent-medium: rgba(201, 169, 110, 0.22);
        --accent-dark: #E6C587;
        --green: #34D399;
        --green-light: rgba(52, 209, 153, 0.1);
        --red-light: rgba(239, 68, 68, 0.1);
        --red: #EF4444;
        --text-primary: #FAFAF8;
        --text-secondary: #C5D1C9;
        --text-muted: #7E8E85;
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.2);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.3);
        --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.4);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
    }
    """
    watch_dial_color = "0x12221A"
    watch_ticks_color = "0xFAFAF8"
    watch_hands_color = "0xFAFAF8"
    plotly_font_color = "#C5D1C9"
    plotly_grid_color = "rgba(255,255,255,0.06)"
    plotly_axis_color = "#7E8E85"
    gauge_num_color = "#FAFAF8"
else:
    theme_variables = """
    :root {
        --bg-primary: #FAFAF8;
        --bg-card: #FFFFFF;
        --bg-card-hover: #FFFFFF;
        --bg-warm: #F5F3EE;
        --border: #E8E5E0;
        --border-hover: #D4CFC7;
        --accent: #C9A96E;
        --accent-light: rgba(201, 169, 110, 0.1);
        --accent-medium: rgba(201, 169, 110, 0.18);
        --accent-dark: #B8943D;
        --green: #2D6A4F;
        --green-light: rgba(45, 106, 79, 0.08);
        --red-light: rgba(220, 53, 69, 0.08);
        --red: #DC3545;
        --text-primary: #1A1A1A;
        --text-secondary: #5A5A5A;
        --text-muted: #9A9A9A;
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.08);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
    }
    """
    watch_dial_color = "0xF5F3EE"
    watch_ticks_color = "0x1A1A1A"
    watch_hands_color = "0x1A1A1A"
    plotly_font_color = "#5A5A5A"
    plotly_grid_color = "#F0EDE8"
    plotly_axis_color = "#9A9A9A"
    gauge_num_color = "#1A1A1A"

# ============================================================
# PROFESSIONAL CSS — Clean SaaS Dashboard
# ============================================================
css_content = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@400;500;600;700&display=swap');

    THEME_VARIABLES_PLACEHOLDER

    /* ── Global ──────────────────────────────── */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    [data-testid="stSidebar"] { display: none; }
    .block-container { max-width: 1200px; padding-top: 90px !important; }
    #MainMenu, footer, header { display: none !important; }

    /* ── Navigation Bar ──────────────────────── */
    .nav-bar {
        position: fixed;
        top: 0; left: 0; width: 100%;
        height: 64px;
        background: var(--bg-card);
        border-bottom: 1px solid var(--border);
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    .nav-container {
        width: 100%;
        max-width: 1200px;
        padding: 0 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .nav-logo, .nav-logo:hover, .nav-logo:active, .nav-logo:focus {
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
        color: var(--text-primary) !important;
        letter-spacing: 0.3px;
        text-decoration: none !important;
    }
    .nav-logo-icon {
        width: 34px;
        height: 34px;
        background: var(--accent);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none !important;
    }
    .nav-logo-icon svg {
        width: 18px;
        height: 18px;
        stroke: #FFFFFF !important;
    }
    .nav-logo-text {
        text-decoration: none !important;
        color: var(--text-primary) !important;
    }
    .nav-logo-text span {
        color: var(--accent);
    }
    .nav-cta, .nav-cta:hover, .nav-cta:active, .nav-cta:focus {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--accent) !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 9px 18px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        text-decoration: none !important;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }
    .nav-cta:hover {
        background: var(--accent-dark) !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }

    /* ── Navigation Tabs ─────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        position: fixed;
        top: 12px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000000;
        gap: 4px;
        background: var(--bg-warm) !important;
        border: 1px solid var(--border) !important;
        border-radius: 100px;
        padding: 4px 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 100px !important;
        padding: 6px 20px !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 25px; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    /* ── Hero Section ────────────────────────── */
    .hero-section {
        padding: 40px 0 20px 0;
        position: relative;
    }
    .brand-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.72rem;
        color: var(--accent);
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }
    .brand-badge::before {
        content: '';
        display: inline-block;
        width: 28px;
        height: 2px;
        background: var(--accent);
        margin-right: 4px;
    }
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        line-height: 1.12;
        margin: 0;
        color: var(--text-primary);
        letter-spacing: -0.5px;
    }
    .hero-title .highlight {
        color: var(--accent);
    }
    .hero-sub {
        color: var(--text-secondary);
        font-size: 1.02rem;
        margin-top: 16px;
        font-weight: 400;
        line-height: 1.65;
    }
    .hero-buttons {
        display: flex;
        gap: 12px;
        margin-top: 28px;
        flex-wrap: wrap;
    }
    .hero-btn-primary, .hero-btn-primary:hover, .hero-btn-primary:active, .hero-btn-primary:focus {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--accent) !important;
        color: #FFFFFF !important;
        padding: 12px 24px !important;
        border-radius: var(--radius-sm) !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    .hero-btn-primary:hover {
        background: var(--accent-dark) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    .hero-btn-secondary, .hero-btn-secondary:hover, .hero-btn-secondary:active, .hero-btn-secondary:focus {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: transparent !important;
        color: var(--text-primary) !important;
        padding: 12px 24px !important;
        border-radius: var(--radius-sm) !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        transition: all 0.2s ease !important;
        border: 1px solid var(--border) !important;
    }
    .hero-btn-secondary:hover {
        border-color: var(--border-hover) !important;
        background: var(--bg-warm) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Clean Card ───────────────────────────── */
    .g-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 24px;
        position: relative;
        overflow: hidden;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    .g-card:hover {
        border-color: var(--border-hover);
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    .card-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* ── KPI Cards ────────────────────────────── */
    .kpi-row {
        display: flex;
        gap: 16px;
        margin: 20px 0;
    }
    .kpi-card {
        flex: 1;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    .kpi-card:hover {
        border-color: var(--border-hover);
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    .kpi-value {
        font-family: 'Poppins', sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-muted);
        font-weight: 500;
        margin-top: 6px;
    }
    .kpi-icon {
        font-size: 1.3rem;
        margin-bottom: 8px;
    }

    /* ── Result Card ──────────────────────────── */
    .result-container {
        background: var(--bg-card);
        border: 1px solid var(--accent);
        border-left: 4px solid var(--accent);
        box-shadow: var(--shadow-md);
        border-radius: var(--radius-md);
        padding: 36px 28px;
        text-align: center;
        margin: 25px 0;
        position: relative;
        overflow: hidden;
    }
    .result-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 8px;
    }
    .result-price {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
        margin: 8px 0;
    }
    .result-range {
        color: var(--text-secondary);
        font-size: 0.92rem;
        font-weight: 400;
        margin-top: 8px;
    }
    .result-range span {
        color: var(--accent);
        font-weight: 600;
    }

    /* ── Metrics Row ──────────────────────────── */
    .metrics-row {
        display: flex;
        gap: 12px;
        margin: 20px 0;
    }
    .metric-pill {
        flex: 1;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 16px;
        text-align: center;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    .metric-pill:hover {
        border-color: var(--border-hover);
        box-shadow: var(--shadow-md);
    }
    .metric-pill .val {
        font-family: 'Poppins', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .metric-pill .lbl {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-muted);
        margin-top: 5px;
        font-weight: 500;
    }

    /* ── Insight Card ─────────────────────────── */
    .insight-card {
        background: var(--bg-warm);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: var(--radius-sm);
        padding: 16px 20px;
        margin: 10px 0;
        display: flex;
        align-items: flex-start;
        gap: 14px;
    }
    .insight-card .icon {
        font-size: 1.2rem;
        margin-top: 1px;
    }
    .insight-card .text {
        color: var(--text-secondary);
        font-size: 0.88rem;
        line-height: 1.65;
    }
    .insight-card .text strong {
        color: var(--accent-dark);
    }

    /* ── Buttons ──────────────────────────────── */
    .stButton > button {
        background: var(--accent) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stButton > button:hover {
        background: var(--accent-dark) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Global Anchor Reset ──────────────────── */
    .stApp a {
        color: var(--accent) !important;
        text-decoration: none !important;
        transition: color 0.2s ease !important;
    }
    .stApp a:hover {
        color: var(--accent-dark) !important;
    }
    .stApp a.nav-cta, 
    .stApp a.nav-cta:hover, 
    .stApp a.nav-cta:active, 
    .stApp a.nav-cta:focus,
    .stApp a.hero-btn-primary, 
    .stApp a.hero-btn-primary:hover, 
    .stApp a.hero-btn-primary:active, 
    .stApp a.hero-btn-primary:focus {
        color: #FFFFFF !important;
    }
    .stApp a.hero-btn-secondary, 
    .stApp a.hero-btn-secondary:hover, 
    .stApp a.hero-btn-secondary:active, 
    .stApp a.hero-btn-secondary:focus {
        color: var(--text-primary) !important;
    }

    /* ── Select Boxes & Inputs ────────────────── */
    .stSelectbox label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }
    .stRadio label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    div[data-testid="stExpander"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
    }
    div[data-testid="stExpander"]:hover {
        border-color: var(--border-hover);
    }

    /* ── Focus and widget overrides (removing default blue) ── */
    div[data-baseweb="select"] > div {
        border-color: var(--border) !important;
    }
    div[data-baseweb="select"]:focus-within > div {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }
    div[role="listbox"] ul li[aria-selected="true"] {
        background-color: var(--accent-light) !important;
        color: var(--text-primary) !important;
    }

    /* ── Section Headers ─────────────────────── */
    .section-hdr {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 25px 0 15px 0;
    }
    .section-hdr .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent);
    }
    .section-hdr h3 {
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }

    /* ── Separator ────────────────────────────── */
    .clean-sep {
        height: 1px;
        margin: 30px 0;
        background: var(--border);
    }

    /* ── Footer ───────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 30px 0 15px 0;
        margin-top: 50px;
        border-top: 1px solid var(--border);
    }
    .app-footer p {
        color: var(--text-muted);
        font-size: 0.82rem;
        letter-spacing: 0.3px;
    }
    .app-footer a {
        color: var(--accent) !important;
        text-decoration: none;
        font-weight: 500;
    }
    .app-footer a:hover {
        color: var(--accent-dark) !important;
    }

    /* ── Streamlit Overrides ──────────────────── */
    .stMarkdown p, .stMarkdown li {
        color: var(--text-secondary);
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-primary);
        font-family: 'Poppins', sans-serif !important;
    }

    /* ── Mobile Responsiveness ───────────────── */
    @media (max-width: 768px) {
        .nav-logo-text {
            display: none !important;
        }
        .nav-cta span {
            display: none !important;
        }
        .nav-cta {
            padding: 8px !important;
            border-radius: 50% !important;
            min-width: 36px;
            min-height: 36px;
            justify-content: center;
        }
        .stTabs [data-baseweb="tab-list"] {
            top: 14px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 14px !important;
            font-size: 0.78rem !important;
        }
        .hero-section {
            padding: 20px 0 !important;
        }
        .hero-title {
            font-size: 2.2rem !important;
        }
        .hero-sub {
            font-size: 0.9rem !important;
            max-width: 100% !important;
        }
        .hero-buttons {
            flex-direction: column !important;
        }
        .hero-btn-primary, .hero-btn-secondary {
            width: 100% !important;
            justify-content: center !important;
        }
        .kpi-row {
            flex-direction: column !important;
            gap: 10px !important;
        }
        .metrics-row {
            flex-direction: column !important;
            gap: 8px !important;
        }
        .result-price {
            font-size: 2.2rem !important;
        }
        .g-card {
            padding: 18px !important;
        }
        div[data-testid="stToggle"] {
            right: 80px !important;
            top: 15px !important;
        }
    }
    @media (max-width: 480px) {
        .stTabs [data-baseweb="tab"] {
            padding: 4px 10px !important;
            font-size: 0.72rem !important;
        }
        .hero-title {
            font-size: 1.8rem !important;
        }
        .kpi-value {
            font-size: 1.3rem !important;
        }
    }

    /* ── Floating Theme Toggle ────────────────── */
    div[data-testid="stToggle"] {
        position: fixed;
        top: 15px;
        right: 140px;
        z-index: 10000000;
        background: transparent !important;
        border: none !important;
        width: auto !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    div[data-testid="stToggle"]::before {
        content: '🌓';
        font-size: 1.1rem;
        margin-right: 8px;
        opacity: 0.8;
    }

    /* ── BaseWeb Dropdowns & Selectboxes Dark Mode Fixes ── */
    div[data-baseweb="select"] {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    div[data-baseweb="select"] div {
        color: var(--text-primary) !important;
    }
    div[data-baseweb="select"] svg {
        fill: var(--text-secondary) !important;
    }
    div[data-baseweb="popover"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
    }
    div[data-baseweb="popover"] li {
        color: var(--text-primary) !important;
        background-color: var(--bg-card) !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: var(--accent-light) !important;
    }
    div[data-baseweb="popover"] li[aria-selected="true"] {
        background-color: var(--accent-medium) !important;
    }
    div[data-testid="stExpander"] summary {
        color: var(--text-primary) !important;
    }
    div[data-testid="stExpander"] summary svg {
        fill: var(--text-secondary) !important;
    }
</style>
""".replace("THEME_VARIABLES_PLACEHOLDER", theme_variables)
st.markdown(css_content, unsafe_allow_html=True)


# ── Custom Navigation Bar ──────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-container">
        <a href="#" class="nav-logo">
            <span class="nav-logo-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <polyline points="12 6 12 12 16 14"/>
                </svg>
            </span>
            <span class="nav-logo-text">Chronos <span>AI</span></span>
        </a>
        <a href="https://github.com/Rudra2986/watch-price-prediction" target="_blank" class="nav-cta">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
            <span>GitHub</span>
        </a>
    </div>
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

col_hero_left, col_hero_right = st.columns([1.2, 1], gap="large")

with col_hero_left:
    st.markdown(f"""
    <div class="hero-section" style="text-align: left;">
        <div class="brand-badge">MACHINE LEARNING · WATCH VALUATION</div>
        <h1 class="hero-title">Predict Watch<br><span class="highlight">Prices</span> with<br>Precision AI</h1>
        <p class="hero-sub" style="max-width: 480px;">
            Intelligent watch price prediction — trained on 1,488 watches
            from the WatchVine marketplace. Hover or touch to rotate the interactive 3D timepiece.
        </p>
        <div class="hero-buttons">
            <a href="#predict" class="hero-btn-primary">Try Prediction →</a>
            <a href="https://github.com/Rudra2986/watch-price-prediction" target="_blank" class="hero-btn-secondary">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                View on GitHub
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_hero_right:
    # 3D Interactive Watch Component (Three.js) — Professional Theme
    watch_3d_html_raw = """
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
            max-width: 380px;
            height: 340px;
            margin: 10px auto 0 auto;
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

        // Materials — Stylized vector wireframe & flat colors
        const goldMaterial = new THREE.MeshBasicMaterial({
            color: 0xC9A96E,
            wireframe: true,
            transparent: true,
            opacity: 0.6
        });
        const darkGreenMaterial = new THREE.MeshBasicMaterial({
            color: 0x2D6A4F,
            wireframe: true,
            transparent: true,
            opacity: 0.6
        });
        const dialMaterial = new THREE.MeshBasicMaterial({
            color: WATCH_DIAL_COLOR,
            transparent: true,
            opacity: 0.95
        });
        const ticksMaterial = new THREE.MeshBasicMaterial({
            color: WATCH_TICKS_COLOR
        });
        const strapMaterial = new THREE.MeshBasicMaterial({
            color: 0x2D6A4F,
            wireframe: true,
            transparent: true,
            opacity: 0.35
        });

        // Outer Bezel / Case (Wireframe Gold)
        const bezelGeom = new THREE.TorusGeometry(1.6, 0.18, 12, 48);
        const bezel = new THREE.Mesh(bezelGeom, goldMaterial);
        watchGroup.add(bezel);

        // Inner Bezel Ring (Wireframe Green)
        const bezelInnerGeom = new THREE.TorusGeometry(1.42, 0.04, 12, 48);
        const bezelInner = new THREE.Mesh(bezelInnerGeom, darkGreenMaterial);
        watchGroup.add(bezelInner);

        // Dial Face (Flat Soft Beige)
        const dialGeom = new THREE.CylinderGeometry(1.38, 1.38, 0.02, 64);
        dialGeom.rotateX(Math.PI / 2);
        const dial = new THREE.Mesh(dialGeom, dialMaterial);
        watchGroup.add(dial);

        // Hour Ticks (12 clean flat marks - no tiny minute dots)
        const ticksGroup = new THREE.Group();
        for (let i = 0; i < 12; i++) {
            const angle = (i * Math.PI * 2) / 12;
            const isMain = (i % 3 === 0);
            const tickW = isMain ? 0.04 : 0.02;
            const tickH = isMain ? 0.16 : 0.08;
            const tickGeom = new THREE.BoxGeometry(tickW, tickH, 0.02);
            const tick = new THREE.Mesh(tickGeom, ticksMaterial);
            tick.position.x = Math.sin(angle) * 1.24;
            tick.position.y = Math.cos(angle) * 1.24;
            tick.position.z = 0.02;
            tick.rotation.z = -angle;
            ticksGroup.add(tick);
        }
        watchGroup.add(ticksGroup);

        // Strap — Top (Simplified Wireframe)
        const strapTopGeom = new THREE.BoxGeometry(0.7, 1.5, 0.05);
        const strapTop = new THREE.Mesh(strapTopGeom, strapMaterial);
        strapTop.position.y = 2.1;
        strapTop.position.z = -0.1;
        strapTop.rotation.x = -0.1;
        watchGroup.add(strapTop);

        // Strap — Bottom (Simplified Wireframe)
        const strapBottomGeom = new THREE.BoxGeometry(0.7, 1.5, 0.05);
        const strapBottom = new THREE.Mesh(strapBottomGeom, strapMaterial);
        strapBottom.position.y = -2.1;
        strapBottom.position.z = -0.1;
        strapBottom.rotation.x = 0.1;
        watchGroup.add(strapBottom);

        // Crown
        const crownGeom = new THREE.CylinderGeometry(0.14, 0.14, 0.25, 12);
        const crown = new THREE.Mesh(crownGeom, goldMaterial);
        crown.position.x = 1.75;
        crown.rotation.z = -Math.PI / 2;
        watchGroup.add(crown);

        // Center hub
        const hubGeom = new THREE.CylinderGeometry(0.08, 0.08, 0.04, 16);
        hubGeom.rotateX(Math.PI / 2);
        const hub = new THREE.Mesh(hubGeom, goldMaterial);
        hub.position.z = 0.03;
        watchGroup.add(hub);

        // Hour hand (Flat Dark Gray)
        const hourHandGeom = new THREE.BoxGeometry(0.06, 0.7, 0.01);
        hourHandGeom.translate(0, 0.35, 0.02);
        const hourHand = new THREE.Mesh(hourHandGeom, ticksMaterial);
        watchGroup.add(hourHand);

        // Minute hand (Flat Dark Green)
        const minHandGeom = new THREE.BoxGeometry(0.04, 1.0, 0.01);
        minHandGeom.translate(0, 0.5, 0.03);
        const minHand = new THREE.Mesh(minHandGeom, new THREE.MeshBasicMaterial({ color: 0x2D6A4F }));
        watchGroup.add(minHand);

        // Second hand (Flat Champagne Gold)
        const secHandGeom = new THREE.BoxGeometry(0.015, 1.15, 0.005);
        secHandGeom.translate(0, 0.575, 0.04);
        const secHand = new THREE.Mesh(secHandGeom, new THREE.MeshBasicMaterial({ color: 0xC9A96E }));
        watchGroup.add(secHand);

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

            // Real-time clock hands
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

            // Gentle base rotation
            const baseSpin = Date.now() * 0.00015;

            // Mouse tracking + gentle floating
            watchGroup.rotation.y += (targetX * 0.5 + Math.sin(baseSpin) * 0.08 - watchGroup.rotation.y) * 0.04;
            watchGroup.rotation.x += (targetY * 0.3 + Math.cos(baseSpin) * 0.05 - watchGroup.rotation.x) * 0.04;

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
    watch_3d_html = watch_3d_html_raw.replace("WATCH_DIAL_COLOR", watch_dial_color).replace("WATCH_TICKS_COLOR", watch_ticks_color).replace("WATCH_HANDS_COLOR", watch_hands_color)
    import streamlit.components.v1 as components
    components.html(watch_3d_html, height=360)

# ── KPI Strip ────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-value">{r2_val:.1%}</div>
        <div class="kpi-label">Model Accuracy</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">📊</div>
        <div class="kpi-value">1,488</div>
        <div class="kpi-label">Watches Trained</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">⚙️</div>
        <div class="kpi-value">{len(train_cols)}</div>
        <div class="kpi-label">AI Features</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">📐</div>
        <div class="kpi-value">±₹{rmse_val:.0f}</div>
        <div class="kpi-label">Precision</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="clean-sep"></div>', unsafe_allow_html=True)


# ============================================================
# TABS — Predict | Analytics | About
# ============================================================

tab_predict, tab_analytics, tab_about = st.tabs(["Predict", "Analytics", "About"])


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

    # ── Input Form ──────────────────────────────────────────
    st.markdown('<div class="g-card">', unsafe_allow_html=True)
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        category = st.selectbox("Category", category_options)
        brand = st.selectbox("Brand", brand_options)
        watch_type = st.selectbox("Watch Type", watch_type_options)
        style = st.selectbox("Style", style_options)
        is_automatic = st.radio(
            "Movement", ["Unknown", "Automatic", "Non-Automatic"],
            horizontal=True
        )

    with col_r:
        belt_type = st.selectbox("Belt Type", belt_type_options)
        case_material = st.selectbox("Case Material", case_material_options)
        strap_material = st.selectbox("Strap Material", strap_material_options)
        dial_color = st.selectbox("Dial Color", dial_color_options)
        strap_color = st.selectbox("Strap Color", strap_color_options)
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
                <div class="metric-pill" style="border-color: var(--accent); border-left: 3px solid var(--accent);">
                    <div class="val" style="color: var(--accent); font-weight: 700; font-size: 1.15rem;">₹{price:,.0f}</div>
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
                number={"suffix": "%", "font": {"size": 36, "color": gauge_num_color, "family": "Poppins"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": plotly_axis_color,
                             "tickfont": {"color": plotly_axis_color, "size": 10}},
                    "bar": {"color": "#C9A96E", "thickness": 0.3},
                    "bgcolor": "#101F18" if st.session_state.theme == "dark" else "#F5F3EE",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(220, 53, 69, 0.08)"},
                        {"range": [30, 60], "color": "rgba(245, 158, 11, 0.08)"},
                        {"range": [60, 100], "color": "rgba(45, 106, 79, 0.08)"},
                    ],
                    "threshold": {
                        "line": {"color": "#2D6A4F", "width": 3},
                        "thickness": 0.8, "value": r2_val * 100,
                    },
                },
            ))
            gauge.update_layout(
                height=220,
                margin=dict(l=30, r=30, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": plotly_axis_color},
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

                colors = ["#DC3545" if v < 0 else "#2D6A4F" for v in top["SHAP"]]

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
                    font=dict(color=plotly_font_color, size=11, family="Inter"),
                    xaxis=dict(
                        gridcolor=plotly_grid_color,
                        title="Impact on Model Output",
                        title_font=dict(color=plotly_axis_color, size=11),
                        tickfont=dict(color=plotly_axis_color),
                    ),
                    yaxis=dict(
                        gridcolor=plotly_grid_color,
                        tickfont=dict(color=plotly_font_color),
                    ),
                    hoverlabel=dict(
                        bgcolor="#12221A" if st.session_state.theme == "dark" else "#FFFFFF",
                        font_size=12,
                        bordercolor=plotly_grid_color,
                        font_color=plotly_font_color,
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="insight-card">
                    <div class="icon">💡</div>
                    <div class="text">
                        <strong style="color:#2D6A4F;">Green bars</strong> push the price UP.
                        <strong style="color:#DC3545;">Red bars</strong> push the price DOWN.
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
            <div style="font-size:3.5rem; margin-bottom:15px; opacity:0.25;">⌚</div>
            <p style="color:var(--text-secondary); font-size:1.05rem; font-weight:500;">
                Configure watch attributes above
            </p>
            <p style="color:var(--text-muted); font-size:0.88rem; margin-top:5px;">
                Click <strong style="color:var(--accent);">Generate Price Prediction</strong> to get an AI-powered estimate
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

    # ── Row 1: KPI Cards + Breakthrough Details ──────────────
    col_kpi_l, col_kpi_r = st.columns([1.8, 1.2], gap="large")

    with col_kpi_l:
        rmse_log = metadata.get("rmse_log", 0.178)
        st.markdown(f"""
        <div class="kpi-row" style="margin: 0; display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px;">
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
        <div class="g-card" style="height: 100%; border-left: 3px solid var(--accent);">
            <div class="card-label">Breakthrough Insight</div>
            <p style="color:var(--text-primary); font-size:1.1rem; font-weight:600; margin:8px 0 12px 0; font-family:'Poppins',sans-serif;">
                Title Feature Extraction
            </p>
            <p style="color:var(--text-secondary); font-size:0.85rem; line-height:1.7; margin:0;">
                Resolving name sparsity was the key: watch brand & movement details were extracted directly from titles.
                This consolidated 62 noisy values down to <strong style="color:var(--accent-dark);">49 standardized brands</strong> and recovered 300+ movements,
                boosting the model R² accuracy from <strong style="color:var(--accent-dark);">45% to 71.2%</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="clean-sep"></div>', unsafe_allow_html=True)

    # ── Row 2: Model Comparison Chart + Active Configuration ─
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
        m_colors = ["#C9A96E" if n == model_name else "#E8DCC8" for n in m_names]

        fig_comp = go.Figure(go.Bar(
            x=m_scores, y=m_names, orientation="h",
            marker=dict(color=m_colors, cornerradius=4,
                        line=dict(width=0)),
            text=[f"{s:.4f}" for s in m_scores],
            textposition="outside",
            textfont=dict(color="#5A5A5A", size=12,
                          family="Inter"),
            hovertemplate="<b>%{y}</b><br>R² = %{x:.4f}<extra></extra>",
        ))
        fig_comp.update_layout(
            height=320,
            margin=dict(l=10, r=60, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=plotly_font_color, size=12, family="Inter"),
            xaxis=dict(
                gridcolor=plotly_grid_color,
                range=[0, 0.85],
                title="R² Score (higher is better)",
                title_font=dict(color=plotly_axis_color, size=11),
                tickfont=dict(color=plotly_axis_color),
            ),
            yaxis=dict(
                gridcolor=plotly_grid_color,
                tickfont=dict(color=plotly_font_color),
            ),
            hoverlabel=dict(
                bgcolor="#12221A" if st.session_state.theme == "dark" else "#FFFFFF",
                bordercolor=plotly_grid_color,
                font_color=plotly_font_color,
            ),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart_r:
        st.markdown(f"""
        <div class="g-card" style="height: 100%;">
            <div class="card-label">Active Model Config</div>
            <p style="color:var(--text-primary); font-size:1.05rem; font-weight:600; margin:8px 0; font-family:'Poppins',sans-serif;">
                {model_name} Predictor
            </p>
            <ul style="color:var(--text-secondary); font-size:0.82rem; line-height:2; padding-left:16px; margin-top:12px;">
                <li>Training Features: <strong style="color:var(--text-primary);">{len(train_cols)}</strong></li>
                <li>Hyperparameter Tuning: <strong style="color:var(--text-primary);">Optuna</strong></li>
                <li>Trials Evaluated: <strong style="color:var(--text-primary);">100 Trials</strong></li>
                <li>Loss Metric: <strong style="color:var(--text-primary);">Root Mean Squared Error</strong></li>
                <li>Target Scale: <strong style="color:var(--text-primary);">Log-transformed</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="clean-sep"></div>', unsafe_allow_html=True)

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
            <div style="font-size:1.8rem; margin-bottom:10px;">{icon}</div>
            <div style="color:var(--text-primary); font-weight:600; font-size:0.9rem; font-family:'Poppins',sans-serif;">{title}</div>
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
            automatic watches. This boosted R² model accuracy from 45% to 71.2%.
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
            <p style="color:var(--text-primary); font-size:1.1rem; font-weight:600; margin:5px 0; font-family:'Poppins',sans-serif;">
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
            <p style="color:var(--text-primary); font-weight:600; margin:5px 0; font-family:'Poppins',sans-serif;">
                Rudra
            </p>
            <p style="color:var(--text-secondary); font-size:0.88rem;">
                GitHub: <a href="https://github.com/Rudra2986" target="_blank"
                style="color:var(--accent); font-weight:500;">@Rudra2986</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="g-card">
            <div class="card-label">Tech Stack</div>
            <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:12px;">
        """ + "".join([
            f'<span style="background:var(--accent-light); border:1px solid rgba(201,169,110,0.25); border-radius:6px; padding:5px 12px; font-size:0.78rem; color:var(--accent-dark); font-weight:600;">{t}</span>'
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
            <ul style="color:var(--text-secondary); font-size:0.85rem; line-height:2; padding-left:18px; margin-top:10px;">
                <li>'Unknown' kept as valid category — not imputed</li>
                <li>Train/test split <strong style="color:var(--text-primary);">before</strong> encoding (leakage-free)</li>
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
