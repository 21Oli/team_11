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
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

REGIONS = [
    "Amhara",
    "Oromia",
    "SNNPR",
    "Somali",
    "Tigray",
    "Afar",
    "Benishangul-Gumuz",
    "Gambela",
    "Harari",
    "Addis Ababa",
    "Dire Dawa",
]

CROPS = [
    "barley",
    "maize",
    "sorghum",
    "wheat",
    "teff",
]

# Features used by the trained model.
# plot_id, price and target are intentionally excluded.
MODEL_FEATURES = [
    "region",
    "crop_type",
    "survey_year",
    "planting_month",
    "altitude_m",
    "rainfall_mm_season",
    "farm_size_ha",
    "fertilizer_kg_per_ha",
    "improved_seed_used",
    "pest_disease_flag",
    "soil_quality_index",
    "labor_days_per_ha",
    "distance_to_market_km",
    "season_mean_temp_c",
    "season_rainfall_total_mm",
    "season_extreme_heat_days",
    "season_temp_std_c",
    "season_temp_deviation_c",
    "weather_months_available",
    "weather_months_expected",
    "weather_months_missing",
    "planting_month_num",
    "fertilizer_improved_seed_interaction",
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
price_df = load_reference_table(PRICE_PATHS)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_divide(a, b):
    """Safely calculate a / b."""
    if b is None or b == 0:
        return 0.0
    return float(a) / float(b)


def find_column(df, candidates):
    """Find the first matching column from a list of possible names."""
    if df is None:
        return None

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def get_reference_price(region, crop, year):
    """
    Try to find a reference price from the bundled price table.

    This price is displayed only as economic context.
    It is NEVER passed to the ML model.
    """
    if price_df is None or price_df.empty:
        return None

    region_col = find_column(
        price_df,
        ["region"]
    )

    crop_col = find_column(
        price_df,
        ["crop_type", "crop"]
    )

    year_col = find_column(
        price_df,
        ["survey_year", "year"]
    )

    price_col = find_column(
        price_df,
        ["price_birr_per_quintal", "price", "price_birr"]
    )

    if not all([region_col, crop_col, year_col, price_col]):
        return None

    data = price_df.copy()

    data["_year"] = pd.to_numeric(
        data[year_col],
        errors="coerce"
    )

    match = data[
        (data[region_col].astype(str).str.lower() == region.lower())
        & (data[crop_col].astype(str).str.lower() == crop.lower())
        & (data["_year"] == int(year))
    ]

    if match.empty:
        # Fall back to crop + region without year
        match = data[
            (data[region_col].astype(str).str.lower() == region.lower())
            & (data[crop_col].astype(str).str.lower() == crop.lower())
        ]

    if match.empty:
        return None

    values = pd.to_numeric(
        match[price_col],
        errors="coerce"
    ).dropna()

    if values.empty:
        return None

    return float(values.median())


def validate_features(df):
    """Check that the prediction dataframe has the expected features."""
    missing = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(f"- {x}" for x in missing)
        )


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🌾 Ethiopian Crop Yield Predictor")

st.markdown(
    """
    **AI-powered agricultural yield prediction using historical
    farm, crop and weather data.**

    Enter the farm and seasonal conditions below to estimate
    expected crop yield in **tons per hectare**.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🌱 Prediction Settings")

    region = st.selectbox(
        "Region",
        REGIONS,
        index=0,
    )

    crop_type = st.selectbox(
        "Crop type",
        CROPS,
        index=0,
    )

    survey_year = st.number_input(
        "Survey year",
        min_value=2000,
        max_value=2100,
        value=2024,
        step=1,
    )

    planting_month = st.selectbox(
        "Planting month",
        list(MONTH_MAP.keys()),
        index=5,
    )

    st.divider()

    st.caption(
        "The trained model excludes market price from prediction "
        "features to avoid price-driven target leakage."
    )


# ============================================================
# INPUT SECTIONS
# ============================================================

left, right = st.columns(2)


# ------------------------------------------------------------
# FARM INFORMATION
# ------------------------------------------------------------

with left:
    st.subheader("🚜 Farm Information")

    altitude_m = st.number_input(
        "Altitude (m)",
        min_value=0.0,
        max_value=5000.0,
        value=1500.0,
        step=10.0,
    )

    farm_size_ha = st.number_input(
        "Farm size (ha)",
        min_value=0.01,
        max_value=1000.0,
        value=1.0,
        step=0.1,
    )

    fertilizer_kg_per_ha = st.number_input(
        "Fertilizer (kg/ha)",
        min_value=0.0,
        max_value=1000.0,
        value=30.0,
        step=1.0,
    )

    improved_seed_used = st.selectbox(
        "Improved seed used?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

    pest_disease_flag = st.selectbox(
        "Pest / disease detected?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

    soil_quality_index = st.slider(
        "Soil quality index",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    labor_days_per_ha = st.number_input(
        "Labor days/ha",
        min_value=0.0,
        max_value=500.0,
        value=40.0,
        step=1.0,
    )

    distance_to_market_km = st.number_input(
        "Distance to market (km)",
        min_value=0.0,
        max_value=500.0,
        value=10.0,
        step=0.5,
    )


# ------------------------------------------------------------
# WEATHER INFORMATION
# ------------------------------------------------------------

with right:
    st.subheader("🌦️ Seasonal Weather")

    rainfall_mm_season = st.number_input(
        "Seasonal rainfall (mm)",
        min_value=0.0,
        max_value=5000.0,
        value=900.0,
        step=10.0,
    )

    season_mean_temp_c = st.number_input(
        "Season mean temperature (°C)",
        min_value=-10.0,
        max_value=50.0,
        value=18.0,
        step=0.1,
    )

    season_rainfall_total_mm = st.number_input(
        "Season rainfall total (mm)",
        min_value=0.0,
        max_value=5000.0,
        value=900.0,
        step=10.0,
    )

    season_extreme_heat_days = st.number_input(
        "Extreme heat days",
        min_value=0,
        max_value=200,
        value=0,
        step=1,
    )

    season_temp_std_c = st.number_input(
        "Season temperature std (°C)",
        min_value=0.0,
        max_value=20.0,
        value=1.5,
        step=0.1,
    )

    season_temp_deviation_c = st.number_input(
        "Season temperature deviation (°C)",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
    )

    weather_months_expected = st.number_input(
        "Expected weather months",
        min_value=1,
        max_value=12,
        value=4,
        step=1,
    )

    weather_months_available = st.number_input(
        "Available weather months",
        min_value=0,
        max_value=12,
        value=4,
        step=1,
    )

    weather_months_missing = max(
        int(weather_months_expected) - int(weather_months_available),
        0,
    )

    st.info(
        f"Weather months missing: **{weather_months_missing}**"
    )


# ============================================================
# ENGINEERED FEATURES
# ============================================================

planting_month_num = MONTH_MAP[planting_month]

fertilizer_improved_seed_interaction = (
    fertilizer_kg_per_ha * improved_seed_used
)

rainfall_per_fertilizer = safe_divide(
    rainfall_mm_season,
    fertilizer_kg_per_ha,
)


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🌾 Predict Crop Yield",
    type="primary",
    use_container_width=True,
)


if predict_button:

    try:
        model = load_model()

        # ----------------------------------------------------
        # Build exactly the model input schema
        # ----------------------------------------------------

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
            "fertilizer_improved_seed_interaction":
                fertilizer_improved_seed_interaction,
            "rainfall_per_fertilizer": rainfall_per_fertilizer,
        }

        input_df = pd.DataFrame([input_data])

        validate_features(input_df)

        # Keep the exact expected feature order.
        input_df = input_df[MODEL_FEATURES]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(input_df)

        predicted_yield = float(np.asarray(prediction).ravel()[0])

        # Prevent negative display values caused by model output.
        predicted_yield_display = max(predicted_yield, 0.0)

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.success("Prediction completed successfully.")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Predicted Yield",
                f"{predicted_yield_display:.2f} t/ha",
            )

        with result_col2:
            estimated_total = (
                predicted_yield_display * farm_size_ha
            )

            st.metric(
                "Estimated Farm Production",
                f"{estimated_total:.2f} tons",
            )

        with result_col3:
            reference_price = get_reference_price(
                region,
                crop_type,
                survey_year,
            )

            if reference_price is not None:
                st.metric(
                    "Reference Market Price",
                    f"{reference_price:,.0f} Birr/quintal",
                )
            else:
                st.metric(
                    "Reference Market Price",
                    "N/A",
                )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.subheader("📊 Prediction Summary")

        summary_df = pd.DataFrame(
            {
                "Input": [
                    "Region",
                    "Crop",
                    "Survey year",
                    "Planting month",
                    "Farm size",
                    "Seasonal rainfall",
                    "Fertilizer",
                    "Improved seed",
                    "Pest / disease",
                    "Soil quality",
                ],
                "Value": [
                    region,
                    crop_type,
                    int(survey_year),
                    planting_month,
                    f"{farm_size_ha:.2f} ha",
                    f"{rainfall_mm_season:.1f} mm",
                    f"{fertilizer_kg_per_ha:.1f} kg/ha",
                    "Yes" if improved_seed_used else "No",
                    "Yes" if pest_disease_flag else "No",
                    f"{soil_quality_index:.2f}",
                ],
            }
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # IMPORTANT MODEL NOTE
        # ----------------------------------------------------

        with st.expander("ℹ️ About this prediction"):
            st.markdown(
                """
                **Target:** `yield_tons_per_ha`

                The prediction uses historical agricultural,
                farm-management and weather-related features.

                The following fields are intentionally **not**
                used as model inputs:

                - `plot_id`
                - `price_birr_per_quintal`
                - `yield_tons_per_ha`

                Market price shown above is reference information
                only and is not passed into the prediction model.
                """
            )

    except Exception as exc:
        st.error("Prediction failed.")

        st.code(
            str(exc),
            language="text",
        )

        st.info(
            "Make sure the validated model is copied to "
            "`app/assets/final_model.joblib` and that the app "
            "uses the same feature schema as the training pipeline."
        )


# ============================================================
# REFERENCE DATA
# ============================================================

with st.expander("📁 Bundled Dataset Information"):

    ref_col1, ref_col2 = st.columns(2)

    with ref_col1:
        st.write("**Weather reference table**")

        if weather_df is not None:
            st.write(
                f"Rows: **{len(weather_df):,}**"
            )
            st.write(
                f"Columns: **{len(weather_df.columns)}**"
            )
        else:
            st.warning(
                "Weather reference table was not found."
            )

    with ref_col2:
        st.write("**Price reference table**")

        if price_df is not None:
            st.write(
                f"Rows: **{len(price_df):,}**"
            )
            st.write(
                f"Columns: **{len(price_df.columns)}**"
            )
        else:
            st.warning(
                "Price reference table was not found."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI & Data Engineering Hackathon • Historical Agricultural "
    "Yield Prediction • Ethiopian Agriculture"
)