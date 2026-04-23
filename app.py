import streamlit as st
import pickle
import numpy as np

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="UAE Rental Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ============================================================
# Custom CSS — warm luxury Dubai theme
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

    * { font-family: 'Outfit', sans-serif; }

    .stApp {
        background: #0b0f19;
    }

    /* Top hero banner */
    .hero-section {
        background: linear-gradient(160deg, #1b1f2e 0%, #141824 100%);
        border: 1px solid rgba(212,175,55,0.1);
        border-radius: 24px;
        padding: 2.5rem 2rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #d4af37, #f5d076, #d4af37, transparent);
    }
    .hero-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #f5f0e8;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }
    .hero-title span {
        background: linear-gradient(135deg, #d4af37, #f5d076);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        color: #6b7280;
        font-size: 1rem;
        font-weight: 400;
    }

    /* Input card sections */
    .input-card {
        background: #141824;
        border: 1px solid #1e2433;
        border-radius: 18px;
        padding: 1.8rem;
        margin-bottom: 1rem;
    }
    .input-card-title {
        color: #d4af37;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid #1e2433;
    }

    /* Streamlit overrides */
    .stSelectbox > div > div { background-color: #0f1320; border-color: #1e2433; border-radius: 10px; }
    .stNumberInput > div > div > input { background-color: #0f1320; border-color: #1e2433; border-radius: 10px; }
    label { color: #9ca3af !important; font-weight: 500 !important; font-size: 0.92rem !important; }
    .stSelectbox label, .stNumberInput label { color: #9ca3af !important; }

    /* Predict button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #d4af37 0%, #b8942e 100%);
        color: #0b0f19;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(212,175,55,0.25);
        background: linear-gradient(135deg, #f5d076 0%, #d4af37 100%);
    }

    /* Result card */
    .result-card {
        background: linear-gradient(160deg, #1b1f2e 0%, #141824 100%);
        border: 1px solid rgba(212,175,55,0.2);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 1rem;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #d4af37, #f5d076, #d4af37, transparent);
    }
    .result-tag {
        display: inline-block;
        background: rgba(212,175,55,0.12);
        color: #d4af37;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }
    .result-yearly {
        font-size: 3.2rem;
        font-weight: 900;
        color: #f5f0e8;
        margin: 0.5rem 0;
        letter-spacing: -1px;
    }
    .result-monthly {
        font-size: 1.4rem;
        font-weight: 600;
        background: linear-gradient(135deg, #d4af37, #f5d076);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .result-divider {
        width: 60px;
        height: 2px;
        background: rgba(212,175,55,0.3);
        margin: 1.2rem auto;
    }

    /* Summary chips */
    .summary-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin-top: 1rem;
    }
    .summary-chip {
        background: rgba(212,175,55,0.08);
        border: 1px solid rgba(212,175,55,0.12);
        color: #9ca3af;
        padding: 0.4rem 0.9rem;
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* Footer */
    .footer-bar {
        text-align: center;
        color: #3b3f4f;
        font-size: 0.75rem;
        padding: 2rem 0 1rem;
        letter-spacing: 0.3px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load model
# ============================================================
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    artifacts = load_model()
    model = artifacts['model']
    scaler = artifacts['scaler']
    type_list = artifacts['type_list']
    furnishing_list = artifacts['furnishing_list']
    location_list = artifacts['location_list']
    city_list = artifacts['city_list']
    model_name = artifacts.get('model_name', 'Unknown')
    r2 = artifacts.get('r2_score', 0)
    num_samples = artifacts.get('num_samples', 0)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Model not found! Run `python train_model.py` first.\n\nError: {e}")

# ============================================================
# Hero Header
# ============================================================
st.markdown("""
<div class="hero-section">
    <div class="hero-emoji">🏠</div>
    <div class="hero-title">UAE <span>Rental Predictor</span></div>
    <div class="hero-sub">AI-powered rental estimates across Dubai, Abu Dhabi & more</div>
</div>
""", unsafe_allow_html=True)

if model_loaded:
    # ============================================================
    # Input Columns — 3-column layout
    # ============================================================
    left_col, mid_col, right_col = st.columns(3)

    with left_col:
        st.markdown('<div class="input-card"><div class="input-card-title">🏗️ Property Details</div></div>', unsafe_allow_html=True)
        property_type = st.selectbox('Property Type', sorted(type_list),
                                     index=sorted(type_list).index('Apartment') if 'Apartment' in type_list else 0)
        area = st.number_input('Area (sqft)', min_value=100, max_value=50000, value=1200, step=50)
        furnishing = st.selectbox('Furnishing', sorted(furnishing_list),
                                  index=sorted(furnishing_list).index('Unfurnished') if 'Unfurnished' in furnishing_list else 0)

    with mid_col:
        st.markdown('<div class="input-card"><div class="input-card-title">🛏️ Room Config</div></div>', unsafe_allow_html=True)
        beds = st.selectbox('Bedrooms', list(range(0, 13)), index=2,
                           help="0 = Studio")
        baths = st.selectbox('Bathrooms', list(range(1, 16)), index=1)

    with right_col:
        st.markdown('<div class="input-card"><div class="input-card-title">📍 Location</div></div>', unsafe_allow_html=True)
        city = st.selectbox('City', sorted(city_list),
                           index=sorted(city_list).index('Dubai') if 'Dubai' in city_list else 0)
        location = st.selectbox('Area / Locality', sorted(location_list), index=0)

    st.write("")  # spacer

    # ============================================================
    # Predict Button
    # ============================================================
    if st.button('💰  Estimate Rental Price'):
        # Encode
        type_enc = type_list.index(property_type) if property_type in type_list else 0
        furn_enc = furnishing_list.index(furnishing) if furnishing in furnishing_list else 0
        loc_enc = location_list.index(location) if location in location_list else location_list.index('Other')
        city_enc = city_list.index(city) if city in city_list else city_list.index('Other')

        features = [[int(beds), int(baths), type_enc, float(area), furn_enc, loc_enc, city_enc]]
        features_scaled = scaler.transform(features)
        prediction = max(model.predict(features_scaled)[0], 0)
        monthly = prediction / 12

        # Format
        if prediction >= 1000000:
            fmt_yearly = f"AED {prediction/1000000:.2f}M"
        else:
            fmt_yearly = f"AED {prediction:,.0f}"
        fmt_monthly = f"AED {monthly:,.0f} / month"

        bed_label = "Studio" if beds == 0 else f"{beds} BHK"

        # Show result
        st.markdown(f"""
        <div class="result-card">
            <div class="result-tag">Estimated Annual Rent</div>
            <div class="result-yearly">{fmt_yearly}</div>
            <div class="result-divider"></div>
            <div class="result-monthly">≈ {fmt_monthly}</div>
            <div class="summary-row">
                <span class="summary-chip">🏢 {bed_label} {property_type}</span>
                <span class="summary-chip">📐 {area:,} sqft</span>
                <span class="summary-chip">🪑 {furnishing}</span>
                <span class="summary-chip">🚿 {baths} Bath</span>
                <span class="summary-chip">📍 {location}</span>
                <span class="summary-chip">🌆 {city}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # Footer
    # ============================================================
    st.markdown(f"""
    <div class="footer-bar">
        Model: {model_name} · {num_samples:,} properties · R² = {r2:.3f} · 7 features<br>
        Built for UAE real estate market analysis
    </div>
    """, unsafe_allow_html=True)
