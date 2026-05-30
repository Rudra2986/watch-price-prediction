# ============================================================
# STEP 7 — STREAMLIT DEPLOYMENT
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
    page_title="Watch Price Predictor",
    page_icon="⌚",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .price-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        margin: 20px 0;
    }
    .price-box h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
    }
    .price-box p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 5px 0 0 0;
    }
    .range-box {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin: 10px 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-card h3 {
        margin: 0;
        color: #667eea;
        font-size: 1.5rem;
    }
    .metric-card p {
        margin: 5px 0 0 0;
        color: #666;
        font-size: 0.85rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9ff 0%, #eef1ff 100%);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL & ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    """Load model and preprocessing artifacts once."""
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

st.markdown('<h1 class="main-title">⌚ Watch Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">WatchVine Dataset · Machine Learning Model</p>', unsafe_allow_html=True)


# ============================================================
# SIDEBAR — MODEL INFO
# ============================================================

with st.sidebar:
    st.markdown("### 📊 Model Info")

    model_name = metadata.get("model_name", "Unknown")
    r2_score_val = metadata.get("r2", 0)
    rmse_rs = metadata.get("rmse_rs", 0)

    st.metric("Model", model_name)
    st.metric("R² Score", f"{r2_score_val:.4f}")
    st.metric("RMSE", f"₹{rmse_rs:.0f}")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app predicts watch prices using a machine learning model
    trained on the WatchVine e-commerce dataset.

    **Dataset**: 1,488 watches  
    **Features**: 96 engineered features  
    **Author**: Rudra  
    """)

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[GitHub Repository](https://github.com/Rudra2986/watch-price-prediction)")


# ============================================================
# INPUT FORM
# ============================================================

st.markdown("### 📝 Enter Watch Details")

col1, col2, col3 = st.columns(3)

with col1:
    category = st.selectbox("Category", ["men's watch", "women's watch"])

    brand = st.selectbox("Brand", [
        "unknown", "audemars piguet", "casio", "citizen", "franck muller",
        "hublot", "iwc", "maxima", "michael kors", "patek philippe",
        "richard mille", "seiko", "tag heuer", "tommy hilfiger", "versace"
    ])

    gender = st.selectbox("Gender", ["men", "women", "unknown"])

    belt_type = st.selectbox("Belt Type", [
        "unknown", "leather_belt", "metal_belt", "rubber_belt", "other"
    ])

    watch_type = st.selectbox("Watch Type", [
        "unknown", "casual", "diving", "dress", "fashion",
        "luxury", "professional", "racing", "sports"
    ])

with col2:
    watch_style_category = st.selectbox("Watch Style Category", [
        "unknown", "casual", "diving", "dress", "fashion",
        "luxury", "professional", "racing", "sports"
    ])

    style = st.selectbox("Style", [
        "unknown", "analog", "analog-digital", "analog/digital",
        "formal", "hybrid", "luxury", "minimalistic",
        "smartwatch", "sporty", "vintage"
    ])

    case_material = st.selectbox("Case Material", [
        "unknown", "carbon fiber", "ceramic", "gold",
        "gold and stainless steel", "metal", "plastic",
        "rose gold", "stainless steel", "stainless steel and gold",
        "stainless steel and rose gold", "titanium"
    ])

    material = st.selectbox("Material", [
        "unknown", "ceramic", "fabric", "gold", "leather", "metal",
        "plastic", "rubber", "silicone", "silicone/rubber",
        "silver", "stainless steel", "titanium"
    ])

    strap_material = st.selectbox("Strap Material", [
        "unknown", "ceramic", "fabric", "leather", "metal",
        "plastic", "rubber", "silicone", "silicone/rubber",
        "stainless steel"
    ])

with col3:
    is_automatic = st.selectbox("Is Automatic?", ["Unknown", "Yes", "No"])

    color = st.selectbox("Watch Color", [
        "unknown", "black", "blue", "brown", "gold", "green",
        "grey", "red", "silver", "white", "rose gold",
        "navy", "orange", "pink", "purple", "beige", "champagne"
    ])

    dial_color = st.selectbox("Dial Color", [
        "unknown", "black", "blue", "brown", "gold", "green",
        "grey", "red", "silver", "white", "rose gold",
        "navy", "orange", "pink", "purple", "beige", "champagne",
        "skeleton"
    ])

    strap_color = st.selectbox("Strap Color", [
        "unknown", "black", "blue", "brown", "gold", "green",
        "grey", "red", "silver", "white", "rose gold",
        "navy", "orange", "pink", "two-tone"
    ])


# ============================================================
# PREPROCESSING — Replicate Step 3 Exactly
# ============================================================

def preprocess_input(category, brand, gender, belt_type, watch_type,
                     watch_style_category, style, case_material, material,
                     strap_material, is_automatic, color, dial_color, strap_color):
    """
    Replicate Step 3 transformations exactly:
    1. Create flag columns
    2. Encode is_automatic
    3. Target encode high-cardinality columns
    4. One-hot encode remaining categoricals
    5. Align with training columns
    """

    # ── Flag columns ─────────────────────────────────────────
    has_brand = 1 if brand.lower() != "unknown" else 0
    has_details = 1 if color.lower() != "unknown" else 0
    has_watch_type = 1 if watch_type.lower() != "unknown" else 0

    # ── is_automatic encoding ────────────────────────────────
    auto_map = {"yes": 1, "no": 0, "unknown": -1}
    is_auto_val = auto_map.get(is_automatic.lower(), -1)

    # ── Target encode color, dial_color, strap_color ─────────
    color_encoded = target_encoders.get("color", {}).get(color, global_mean)
    dial_encoded = target_encoders.get("dial_color", {}).get(dial_color, global_mean)
    strap_c_encoded = target_encoders.get("strap_color", {}).get(strap_color, global_mean)

    # Handle pandas Series .get()
    if hasattr(color_encoded, '__iter__') and not isinstance(color_encoded, str):
        color_encoded = global_mean
    if hasattr(dial_encoded, '__iter__') and not isinstance(dial_encoded, str):
        dial_encoded = global_mean
    if hasattr(strap_c_encoded, '__iter__') and not isinstance(strap_c_encoded, str):
        strap_c_encoded = global_mean

    # ── Build base row ───────────────────────────────────────
    row = {
        "color": color_encoded,
        "dial_color": dial_encoded,
        "strap_color": strap_c_encoded,
        "is_automatic": is_auto_val,
        "has_brand": has_brand,
        "has_details": has_details,
        "has_watch_type": has_watch_type,
    }

    # ── One-hot encode categoricals ──────────────────────────
    ohe_mappings = {
        "category": category,
        "gender": gender,
        "belt_type": belt_type,
        "watch_type": watch_type,
        "watch_style_category": watch_style_category,
        "style": style,
        "case_material": case_material,
        "material": material,
        "strap_material": strap_material,
        "brand": brand,
    }

    for col_name in train_cols:
        if col_name not in row:
            row[col_name] = 0  # default all OHE to 0

    # Set the correct OHE columns to 1
    for prefix, value in ohe_mappings.items():
        ohe_col = f"{prefix}_{value}"
        if ohe_col in row:
            row[ohe_col] = 1

    # Create DataFrame with exact column order
    input_df = pd.DataFrame([row])[train_cols]

    return input_df


# ============================================================
# PREDICT
# ============================================================

st.markdown("---")

if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
    # Preprocess
    input_df = preprocess_input(
        category, brand, gender, belt_type, watch_type,
        watch_style_category, style, case_material, material,
        strap_material, is_automatic, color, dial_color, strap_color
    )

    # Predict
    log_pred = model.predict(input_df)[0]
    predicted_price = np.expm1(log_pred)

    # Confidence range using RMSE
    rmse = metadata.get("rmse_rs", 500)
    price_low = max(predicted_price - rmse, 0)
    price_high = predicted_price + rmse

    # ── Display Results ──────────────────────────────────────
    st.markdown(f"""
    <div class="price-box">
        <p>Predicted Price</p>
        <h1>₹{predicted_price:,.0f}</h1>
        <p>Confidence Range: ₹{price_low:,.0f} – ₹{price_high:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ─────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <h3>₹{predicted_price:,.0f}</h3>
            <p>Point Estimate</p>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-card">
            <h3>₹{price_low:,.0f} – ₹{price_high:,.0f}</h3>
            <p>Confidence Range (±RMSE)</p>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{log_pred:.4f}</h3>
            <p>Log Price (raw model output)</p>
        </div>""", unsafe_allow_html=True)

    # ── Feature Summary ──────────────────────────────────────
    with st.expander("📋 Input Feature Summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    # ── SHAP Explanation (if available) ───────────────────────
    try:
        import shap

        with st.expander("🔍 Feature Importance (SHAP)"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            # Top features by absolute SHAP value
            shap_df = pd.DataFrame({
                "Feature": train_cols,
                "SHAP Value": shap_values[0]
            })
            shap_df["Abs SHAP"] = shap_df["SHAP Value"].abs()
            shap_df = shap_df.sort_values("Abs SHAP", ascending=False).head(15)

            st.bar_chart(
                shap_df.set_index("Feature")["SHAP Value"],
                use_container_width=True
            )

            st.caption("Positive SHAP = pushes price UP, Negative = pushes price DOWN")
    except Exception:
        pass

else:
    st.info("👆 Fill in the watch details above and click **Predict Price** to get a prediction.")


# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Watch Price Predictor · Built by Rudra · Summer Internship Project · "
    "<a href='https://github.com/Rudra2986/watch-price-prediction'>GitHub</a>"
    "</p>",
    unsafe_allow_html=True
)
