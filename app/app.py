from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Ethiopian Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS — EXPERT DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        min-height: 100vh;
    }

    /* ── Main content area ── */
    .main .block-container {
        padding: 2rem 2.5rem 3rem 2.5rem;
        max-width: 1300px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2a3b 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #a8c5da !important;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stCaption p {
        color: #607d8b !important;
        font-size: 0.78rem;
        line-height: 1.6;
    }

    /* ── HERO BANNER ── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        display: flex;
        align-items: center;
        gap: 2rem;
    }
    .hero-icon {
        font-size: 4rem;
        line-height: 1;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .hero-title span {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #8fa8bf;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #43e97b22, #38f9d722);
        border: 1px solid rgba(67,233,123,0.3);
        color: #43e97b;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        display: inline-block;
    }

    /* ── SECTION CARDS ── */
    .section-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        padding: 1.6rem 1.8rem 1.8rem 1.8rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(8px);
        transition: border-color 0.2s;
    }
    .section-card:hover {
        border-color: rgba(255,255,255,0.18);
    }
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.01em;
        margin-bottom: 1.2rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-header-icon {
        font-size: 1.1rem;
    }

    /* ── Input labels ── */
    label, .stSelectbox label, .stNumberInput label,
    .stSlider label, [data-testid="stWidgetLabel"] {
        color: #a8c5da !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* ── Input fields ── */
    .stNumberInput input,
    .stSelectbox select,
    .stTextInput input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stNumberInput input:focus,
    .stSelectbox select:focus {
        border-color: rgba(67,233,123,0.5) !important;
        box-shadow: 0 0 0 3px rgba(67,233,123,0.1) !important;
    }

    /* ── Selectbox dropdown ── */
    [data-testid="stSelectbox"] > div > div {
        background: rgba(13,27,42,0.95) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }

    /* ── Slider ── */
    .stSlider [data-testid="stThumbValue"] {
        color: #43e97b !important;
    }
    .stSlider .st-bx { background: rgba(67,233,123,0.2) !important; }
    .stSlider .st-by { background: #43e97b !important; }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%) !important;
        color: #0d1b2a !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
        padding: 0.75rem 2rem !important;
        height: 3.2rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 24px rgba(67,233,123,0.25) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(67,233,123,0.4) !important;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0) !important;
    }

    /* ── Secondary button ── */
    .stButton > button:not([kind="primary"]) {
        background: rgba(255,255,255,0.06) !important;
        color: #a8c5da !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 14px !important;
        padding: 1.4rem 1.6rem !important;
        backdrop-filter: blur(8px);
    }
    [data-testid="stMetricLabel"] {
        color: #8fa8bf !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* ── Prediction result highlight ── */
    .result-card {
        background: linear-gradient(135deg, rgba(67,233,123,0.08) 0%, rgba(56,249,215,0.05) 100%);
        border: 1px solid rgba(67,233,123,0.25);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    .result-yield {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #43e97b, #38f9d7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .result-unit {
        font-size: 1.2rem;
        color: #8fa8bf;
        font-weight: 400;
        margin-top: 0.4rem;
    }
    .result-label {
        font-size: 0.85rem;
        color: #607d8b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* ── Success / info / warning / error messages ── */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }
    [data-testid="stNotificationContentSuccess"] {
        background: rgba(67,233,123,0.1) !important;
        border-left: 4px solid #43e97b !important;
        color: #a8e6c1 !important;
    }
    [data-testid="stNotificationContentError"] {
        background: rgba(255,82,82,0.1) !important;
        border-left: 4px solid #ff5252 !important;
    }
    [data-testid="stNotificationContentInfo"] {
        background: rgba(56,189,248,0.1) !important;
        border-left: 4px solid #38bdf8 !important;
    }
    [data-testid="stNotificationContentWarning"] {
        background: rgba(250,204,21,0.1) !important;
        border-left: 4px solid #facc15 !important;
    }

    /* ── DataFrame / table ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden;
    }
    .stDataFrame table {
        background: rgba(255,255,255,0.03) !important;
        color: #c9d8e4 !important;
    }
    .stDataFrame th {
        background: rgba(255,255,255,0.07) !important;
        color: #a8c5da !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    }
    .stDataFrame td {
        border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 10px !important;
        color: #a8c5da !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
        color: #8fa8bf !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Sidebar divider ── */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.22); }

    /* ── Step indicator ── */
    .step-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: #8fa8bf;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .step-pill-active {
        background: rgba(67,233,123,0.12);
        border-color: rgba(67,233,123,0.3);
        color: #43e97b;
    }

    /* ── KPI strip ── */
    .kpi-strip {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-item {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #43e97b;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #607d8b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
        margin-top: 0.2rem;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        color: #37474f;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
    }
    .app-footer a { color: #546e7a; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"

MODEL_PATH = ASSETS_DIR / "final_model.joblib"

WEATHER_PATHS = [
    ASSETS_DIR / "cleaned_weather.csv",
    ASSETS_DIR / "weather_cleaned.csv",
    ASSETS_DIR / "weather.csv",
]

PRICE_PATHS = [
    ASSETS_DIR / "cleaned_price.csv",
    ASSETS_DIR / "price_cleaned.csv",
    ASSETS_DIR / "price.csv",
]


# ============================================================
# CONSTANTS
# ============================================================

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

REGIONS = [
    "Amhara", "Oromia", "SNNPR", "Somali", "Tigray",
    "Afar", "Benishangul-Gumuz", "Gambela", "Harari",
    "Addis Ababa", "Dire Dawa",
]

CROPS = ["barley", "maize", "sorghum", "wheat", "teff"]

CROP_ICONS = {
    "barley": "🌾", "maize": "🌽", "sorghum": "🌿",
    "wheat": "🌾", "teff": "🌱",
}

# Features used by the trained model.
MODEL_FEATURES = [
    "region", "crop_type", "survey_year", "planting_month",
    "altitude_m", "rainfall_mm_season", "farm_size_ha",
    "fertilizer_kg_per_ha", "improved_seed_used", "pest_disease_flag",
    "soil_quality_index", "labor_days_per_ha", "distance_to_market_km",
    "season_mean_temp_c", "season_rainfall_total_mm",
    "season_extreme_heat_days", "season_temp_std_c",
    "season_temp_deviation_c", "weather_months_available",
    "weather_months_expected", "weather_months_missing",
    "planting_month_num", "fertilizer_improved_seed_interaction",
    "rainfall_per_fertilizer",
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n\n"
            "Copy models/final_model.joblib into app/assets/."
        )
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD OPTIONAL REFERENCE TABLES
# ============================================================

@st.cache_data
def load_reference_table(paths):
    for path in paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                return None
    return None


weather_df = load_reference_table(WEATHER_PATHS)
price_df   = load_reference_table(PRICE_PATHS)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_divide(a, b):
    if b is None or b == 0:
        return 0.0
    return float(a) / float(b)


def find_column(df, candidates):
    if df is None:
        return None
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def get_reference_price(region, crop, year):
    if price_df is None or price_df.empty:
        return None

    region_col = find_column(price_df, ["region"])
    crop_col   = find_column(price_df, ["crop_type", "crop"])
    year_col   = find_column(price_df, ["survey_year", "year"])
    price_col  = find_column(price_df, ["price_birr_per_quintal", "price", "price_birr"])

    if not all([region_col, crop_col, year_col, price_col]):
        return None

    data = price_df.copy()
    data["_year"] = pd.to_numeric(data[year_col], errors="coerce")

    match = data[
        (data[region_col].astype(str).str.lower() == region.lower())
        & (data[crop_col].astype(str).str.lower() == crop.lower())
        & (data["_year"] == int(year))
    ]

    if match.empty:
        match = data[
            (data[region_col].astype(str).str.lower() == region.lower())
            & (data[crop_col].astype(str).str.lower() == crop.lower())
        ]

    if match.empty:
        return None

    values = pd.to_numeric(match[price_col], errors="coerce").dropna()
    return float(values.median()) if not values.empty else None


def validate_features(df):
    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(f"  • {x}" for x in missing)
        )


def yield_category(value):
    if value < 0.5:
        return "⚠️ Very Low", "#ff5252"
    elif value < 1.5:
        return "🟡 Low", "#facc15"
    elif value < 3.0:
        return "🟢 Moderate", "#43e97b"
    elif value < 5.0:
        return "🔵 Good", "#38bdf8"
    else:
        return "🌟 Excellent", "#a78bfa"


# ============================================================
# HERO BANNER
# ============================================================

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-icon">🌾</div>
        <div>
            <div class="hero-badge">AI · Agricultural Intelligence</div>
            <h1 class="hero-title">Ethiopian <span>Crop Yield</span> Predictor</h1>
            <p class="hero-subtitle">
                Machine-learning powered forecasts using farm management,
                soil quality, and seasonal weather data.
                Enter your conditions to estimate yield in <strong style="color:#c9d8e4">tons per hectare</strong>.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — CONTEXT SETTINGS
# ============================================================

with st.sidebar:
    st.markdown(
        "<h2 style='color:#ffffff;font-size:1.1rem;font-weight:700;"
        "letter-spacing:-0.01em;margin-bottom:0.2rem'>🌍 Context Settings</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#607d8b;font-size:0.78rem;margin-top:0;"
        "margin-bottom:1.4rem'>Location, crop, and season configuration</p>",
        unsafe_allow_html=True,
    )

    region = st.selectbox("Region", REGIONS, index=0)

    crop_type = st.selectbox(
        "Crop Type",
        CROPS,
        index=0,
        format_func=lambda c: f"{CROP_ICONS.get(c, '🌱')}  {c.capitalize()}",
    )

    survey_year = st.number_input(
        "Survey Year",
        min_value=2000,
        max_value=2100,
        value=2024,
        step=1,
    )

    planting_month = st.selectbox(
        "Planting Month",
        list(MONTH_MAP.keys()),
        index=5,
    )

    st.divider()

    # Mini status panel
    planting_month_num = MONTH_MAP[planting_month]
    st.markdown(
        f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
        border-radius:10px;padding:0.9rem 1rem;font-size:0.8rem;color:#8fa8bf;line-height:1.8">
            📍 <b style="color:#c9d8e4">{region}</b><br>
            {CROP_ICONS.get(crop_type,'🌱')} <b style="color:#c9d8e4">{crop_type.capitalize()}</b> &nbsp;·&nbsp; {planting_month} {survey_year}<br>
            📅 Month <b style="color:#43e97b">#{planting_month_num}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption(
        "ℹ️ Market price is shown as economic context only "
        "and is not used by the prediction model to prevent target leakage."
    )


# ============================================================
# MAIN CONTENT — TWO COLUMNS
# ============================================================

col_left, col_right = st.columns([1, 1], gap="large")


# ─────────────────────────────────────────────────────────────
# LEFT — FARM CONDITIONS
# ─────────────────────────────────────────────────────────────

with col_left:
    st.markdown(
        "<div class='section-header'>"
        "<span class='section-header-icon'>🚜</span> Farm Conditions"
        "</div>",
        unsafe_allow_html=True,
    )

    altitude_m = st.number_input(
        "Altitude (m)", min_value=0.0, max_value=5000.0, value=1500.0, step=10.0,
        help="Elevation of the farm above sea level",
    )
    farm_size_ha = st.number_input(
        "Farm Size (ha)", min_value=0.01, max_value=1000.0, value=1.0, step=0.1,
        help="Total cultivated area in hectares",
    )

    c1, c2 = st.columns(2)
    with c1:
        fertilizer_kg_per_ha = st.number_input(
            "Fertilizer (kg/ha)", min_value=0.0, max_value=1000.0, value=30.0, step=1.0,
        )
    with c2:
        labor_days_per_ha = st.number_input(
            "Labor Days / ha", min_value=0.0, max_value=500.0, value=40.0, step=1.0,
        )

    c3, c4 = st.columns(2)
    with c3:
        improved_seed_used = st.selectbox(
            "Improved Seed?",
            options=[0, 1],
            format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
        )
    with c4:
        pest_disease_flag = st.selectbox(
            "Pest / Disease?",
            options=[0, 1],
            format_func=lambda x: "⚠️ Yes" if x == 1 else "✅ No",
        )

    soil_quality_index = st.slider(
        "Soil Quality Index",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="Composite index from 0 (poor) to 1 (excellent)",
    )

    # Visual soil quality bar
    sq_pct = int(soil_quality_index * 100)
    sq_color = "#ff5252" if sq_pct < 30 else "#facc15" if sq_pct < 60 else "#43e97b"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.7rem;margin:-0.5rem 0 0.8rem 0">
            <div style="flex:1;height:6px;background:rgba(255,255,255,0.08);border-radius:3px;overflow:hidden">
                <div style="width:{sq_pct}%;height:100%;background:{sq_color};border-radius:3px;
                transition:width 0.3s"></div>
            </div>
            <span style="font-size:0.8rem;color:{sq_color};font-weight:600;min-width:2.5rem">{sq_pct}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    distance_to_market_km = st.number_input(
        "Distance to Market (km)",
        min_value=0.0, max_value=500.0, value=10.0, step=0.5,
        help="Nearest market distance in kilometres",
    )


# ─────────────────────────────────────────────────────────────
# RIGHT — SEASONAL WEATHER
# ─────────────────────────────────────────────────────────────

with col_right:
    st.markdown(
        "<div class='section-header'>"
        "<span class='section-header-icon'>🌦️</span> Seasonal Weather"
        "</div>",
        unsafe_allow_html=True,
    )

    r1, r2 = st.columns(2)
    with r1:
        rainfall_mm_season = st.number_input(
            "Seasonal Rainfall (mm)", min_value=0.0, max_value=5000.0, value=900.0, step=10.0,
        )
    with r2:
        season_rainfall_total_mm = st.number_input(
            "Rainfall Total (mm)", min_value=0.0, max_value=5000.0, value=900.0, step=10.0,
        )

    r3, r4 = st.columns(2)
    with r3:
        season_mean_temp_c = st.number_input(
            "Mean Temp (°C)", min_value=-10.0, max_value=50.0, value=18.0, step=0.1,
        )
    with r4:
        season_temp_std_c = st.number_input(
            "Temp Std Dev (°C)", min_value=0.0, max_value=20.0, value=1.5, step=0.1,
        )

    r5, r6 = st.columns(2)
    with r5:
        season_temp_deviation_c = st.number_input(
            "Temp Deviation (°C)", min_value=-20.0, max_value=20.0, value=0.0, step=0.1,
        )
    with r6:
        season_extreme_heat_days = st.number_input(
            "Extreme Heat Days", min_value=0, max_value=200, value=0, step=1,
        )

    st.markdown(
        "<div class='section-header' style='margin-top:1rem'>"
        "<span class='section-header-icon'>📅</span> Weather Data Coverage"
        "</div>",
        unsafe_allow_html=True,
    )

    w1, w2 = st.columns(2)
    with w1:
        weather_months_expected = st.number_input(
            "Expected Months", min_value=1, max_value=12, value=4, step=1,
        )
    with w2:
        weather_months_available = st.number_input(
            "Available Months", min_value=0, max_value=12, value=4, step=1,
        )

    weather_months_missing = max(
        int(weather_months_expected) - int(weather_months_available), 0
    )

    # Coverage progress bar
    coverage_pct = (
        int(weather_months_available) / int(weather_months_expected) * 100
        if weather_months_expected > 0 else 0
    )
    cov_color = "#ff5252" if coverage_pct < 50 else "#facc15" if coverage_pct < 80 else "#43e97b"
    cov_label = "Critical" if coverage_pct < 50 else "Partial" if coverage_pct < 80 else "Complete"

    st.markdown(
        f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
        border-radius:10px;padding:0.9rem 1rem;margin-top:0.4rem">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                <span style="font-size:0.78rem;color:#8fa8bf;text-transform:uppercase;
                letter-spacing:0.05em;font-weight:600">Data Coverage</span>
                <span style="font-size:0.85rem;font-weight:700;color:{cov_color}">{cov_label}</span>
            </div>
            <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:4px;overflow:hidden">
                <div style="width:{min(coverage_pct,100):.0f}%;height:100%;background:{cov_color};
                border-radius:4px;transition:width 0.4s"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:0.4rem">
                <span style="font-size:0.75rem;color:#607d8b">
                    {int(weather_months_available)} of {int(weather_months_expected)} months
                </span>
                <span style="font-size:0.75rem;color:#607d8b">
                    {int(weather_months_missing)} missing
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DERIVED / ENGINEERED FEATURES
# ============================================================

fertilizer_improved_seed_interaction = fertilizer_kg_per_ha * improved_seed_used
rainfall_per_fertilizer = safe_divide(rainfall_mm_season, fertilizer_kg_per_ha)


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("<div style='margin: 2rem 0 0.5rem 0'></div>", unsafe_allow_html=True)

predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_button = st.button(
        "⚡  Run Yield Prediction",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# PREDICTION RESULTS
# ============================================================

if predict_button:
    try:
        model = load_model()

        input_data = {
            "region": region,
            "crop_type": crop_type,
            "survey_year": float(survey_year),
            "planting_month": planting_month,
            "altitude_m": altitude_m,
            "rainfall_mm_season": rainfall_mm_season,
            "farm_size_ha": farm_size_ha,
            "fertilizer_kg_per_ha": fertilizer_kg_per_ha,
            "improved_seed_used": improved_seed_used,
            "pest_disease_flag": pest_disease_flag,
            "soil_quality_index": soil_quality_index,
            "labor_days_per_ha": labor_days_per_ha,
            "distance_to_market_km": distance_to_market_km,
            "season_mean_temp_c": season_mean_temp_c,
            "season_rainfall_total_mm": season_rainfall_total_mm,
            "season_extreme_heat_days": season_extreme_heat_days,
            "season_temp_std_c": season_temp_std_c,
            "season_temp_deviation_c": season_temp_deviation_c,
            "weather_months_available": weather_months_available,
            "weather_months_expected": weather_months_expected,
            "weather_months_missing": weather_months_missing,
            "planting_month_num": planting_month_num,
            "fertilizer_improved_seed_interaction": fertilizer_improved_seed_interaction,
            "rainfall_per_fertilizer": rainfall_per_fertilizer,
        }

        input_df = pd.DataFrame([input_data])
        validate_features(input_df)
        input_df = input_df[MODEL_FEATURES]

        prediction = model.predict(input_df)
        predicted_yield = float(np.asarray(prediction).ravel()[0])
        predicted_yield_display = max(predicted_yield, 0.0)
        estimated_total = predicted_yield_display * farm_size_ha

        category_label, category_color = yield_category(predicted_yield_display)
        reference_price = get_reference_price(region, crop_type, survey_year)

        # ── Hero result card ──────────────────────────────────
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted Crop Yield</div>
                <div class="result-yield">{predicted_yield_display:.2f}</div>
                <div class="result-unit">tons per hectare</div>
                <div style="margin-top:0.8rem">
                    <span style="display:inline-block;background:rgba(255,255,255,0.06);
                    border:1px solid rgba(255,255,255,0.1);border-radius:20px;
                    padding:0.3rem 1rem;font-size:0.82rem;font-weight:600;
                    color:{category_color}">{category_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── KPI metrics row ───────────────────────────────────
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("🌾 Yield / ha", f"{predicted_yield_display:.2f} t/ha")
        with m2:
            st.metric("🏡 Farm Production", f"{estimated_total:.2f} tons")
        with m3:
            if reference_price is not None:
                st.metric("💰 Market Price", f"{reference_price:,.0f} Birr/q")
                if reference_price:
                    gross = estimated_total * (reference_price / 10)  # 1 ton = 10 quintals
                    st.caption(f"Gross revenue est. ≈ {gross:,.0f} Birr")
            else:
                st.metric("💰 Market Price", "N/A")
                st.caption("No price data for this selection")

        # ── Input summary table ───────────────────────────────
        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

        with st.expander("📋 Full Input Summary", expanded=False):
            summary_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Region", "Crop Type", "Survey Year", "Planting Month",
                        "Farm Size", "Altitude", "Seasonal Rainfall", "Fertilizer",
                        "Improved Seed", "Pest / Disease", "Soil Quality",
                        "Labor Days / ha", "Distance to Market",
                        "Mean Temperature", "Extreme Heat Days",
                    ],
                    "Value": [
                        region,
                        f"{CROP_ICONS.get(crop_type,'')} {crop_type.capitalize()}",
                        int(survey_year),
                        planting_month,
                        f"{farm_size_ha:.2f} ha",
                        f"{altitude_m:.0f} m",
                        f"{rainfall_mm_season:.1f} mm",
                        f"{fertilizer_kg_per_ha:.1f} kg/ha",
                        "✅ Yes" if improved_seed_used else "❌ No",
                        "⚠️ Yes" if pest_disease_flag else "✅ No",
                        f"{soil_quality_index:.2f} ({sq_pct}%)",
                        f"{labor_days_per_ha:.0f} days",
                        f"{distance_to_market_km:.1f} km",
                        f"{season_mean_temp_c:.1f} °C",
                        f"{season_extreme_heat_days} days",
                    ],
                }
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # ── Model note ────────────────────────────────────────
        with st.expander("ℹ️ About this Prediction", expanded=False):
            st.markdown(
                """
                **Target:** `yield_tons_per_ha`

                The model uses historical agricultural, farm-management,
                and weather-related features.

                **Intentionally excluded** from model inputs to prevent leakage:
                - `plot_id`
                - `price_birr_per_quintal`
                - `yield_tons_per_ha` (the prediction target)

                Market price displayed above is reference information only.
                """
            )

    except Exception as exc:
        st.error("Prediction failed — see details below.")
        st.code(str(exc), language="text")
        st.info(
            "Ensure `app/assets/final_model.joblib` is present and the app "
            "uses the same feature schema as the training pipeline."
        )


# ============================================================
# BUNDLED DATASET INFO
# ============================================================

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

with st.expander("📁 Bundled Dataset Information", expanded=False):
    d1, d2 = st.columns(2)

    with d1:
        st.markdown(
            "<p style='color:#a8c5da;font-weight:600;font-size:0.85rem;"
            "text-transform:uppercase;letter-spacing:0.05em'>🌤 Weather Reference</p>",
            unsafe_allow_html=True,
        )
        if weather_df is not None:
            st.markdown(
                f"<p style='color:#8fa8bf;font-size:0.9rem'>"
                f"<b style='color:#43e97b'>{len(weather_df):,}</b> rows · "
                f"<b style='color:#43e97b'>{len(weather_df.columns)}</b> columns</p>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Weather reference table not found.")

    with d2:
        st.markdown(
            "<p style='color:#a8c5da;font-weight:600;font-size:0.85rem;"
            "text-transform:uppercase;letter-spacing:0.05em'>💰 Price Reference</p>",
            unsafe_allow_html=True,
        )
        if price_df is not None:
            st.markdown(
                f"<p style='color:#8fa8bf;font-size:0.9rem'>"
                f"<b style='color:#43e97b'>{len(price_df):,}</b> rows · "
                f"<b style='color:#43e97b'>{len(price_df.columns)}</b> columns</p>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Price reference table not found.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div class="app-footer">
        🌾 &nbsp; Ethiopian Agricultural Yield Predictor &nbsp;·&nbsp;
        AI &amp; Data Engineering Hackathon &nbsp;·&nbsp;
        Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
