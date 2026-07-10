import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="RideSense-AI",
    page_icon="🚖",
    layout="wide"
)


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate straight-line distance between two geographic points
    using the Haversine formula.

    Returns distance in kilometres.
    """

    earth_radius_km = 6371.0

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    latitude_difference = lat2 - lat1
    longitude_difference = lon2 - lon1

    a = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(longitude_difference / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Approximate Manhattan distance by adding horizontal
    and vertical Haversine distances.
    """

    horizontal_distance = haversine_distance(
        lat1,
        lon1,
        lat1,
        lon2
    )

    vertical_distance = haversine_distance(
        lat1,
        lon1,
        lat2,
        lon1
    )

    return horizontal_distance + vertical_distance


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing from pickup location to dropoff location.
    Returns direction in degrees from 0 to 360.
    """

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    longitude_difference_rad = np.radians(lon2 - lon1)

    x = np.sin(longitude_difference_rad) * np.cos(lat2_rad)

    y = (
        np.cos(lat1_rad) * np.sin(lat2_rad)
        - np.sin(lat1_rad)
        * np.cos(lat2_rad)
        * np.cos(longitude_difference_rad)
    )

    bearing = np.degrees(np.arctan2(x, y))

    return (bearing + 360) % 360


def get_expected_features(trained_model):
    """
    Retrieve the exact feature names used while training.
    Supports XGBoost and standard scikit-learn models.
    """

    if hasattr(trained_model, "get_booster"):
        feature_names = trained_model.get_booster().feature_names

        if feature_names:
            return feature_names

    if hasattr(trained_model, "feature_names_in_"):
        return list(trained_model.feature_names_in_)

    return [
        "vendor_id",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "store_and_fwd_flag",
        "pickup_hour",
        "pickup_day",
        "pickup_month",
        "pickup_weekday",
        "is_weekend",
        "is_rush_hour",
        "is_night",
        "haversine_distance_km",
        "manhattan_distance_km",
        "longitude_difference",
        "latitude_difference",
        "bearing"
    ]


# ---------------------------------------------------------
# Application Header
# ---------------------------------------------------------

st.title("🚖 RideSense-AI")
st.subheader("NYC Taxi Trip Duration Prediction")

st.write(
    "Predict the approximate duration of a New York City taxi trip "
    "using pickup and dropoff coordinates, pickup time, vendor details, "
    "passenger count, and engineered geographic features."
)

st.divider()


# ---------------------------------------------------------
# User Input Section
# ---------------------------------------------------------

st.header("🤖 Predict Trip Duration")

left_column, right_column = st.columns(2)

with left_column:
    vendor_id = st.selectbox(
        "Vendor ID",
        options=[1, 2]
    )

    passenger_count = st.number_input(
        "Passenger Count",
        min_value=1,
        max_value=9,
        value=1,
        step=1
    )

    pickup_hour = st.slider(
        "Pickup Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    pickup_day = st.slider(
        "Pickup Day",
        min_value=1,
        max_value=31,
        value=15
    )

    pickup_month = st.slider(
        "Pickup Month",
        min_value=1,
        max_value=12,
        value=6
    )

    pickup_weekday = st.selectbox(
        "Pickup Weekday",
        options=[
            0, 1, 2, 3, 4, 5, 6
        ],
        format_func=lambda day: {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday"
        }[day]
    )

with right_column:
    pickup_longitude = st.number_input(
        "Pickup Longitude",
        min_value=-75.0,
        max_value=-73.0,
        value=-73.9851,
        format="%.6f"
    )

    pickup_latitude = st.number_input(
        "Pickup Latitude",
        min_value=40.0,
        max_value=42.0,
        value=40.7589,
        format="%.6f"
    )

    dropoff_longitude = st.number_input(
        "Dropoff Longitude",
        min_value=-75.0,
        max_value=-73.0,
        value=-73.9855,
        format="%.6f"
    )

    dropoff_latitude = st.number_input(
        "Dropoff Latitude",
        min_value=40.0,
        max_value=42.0,
        value=40.7484,
        format="%.6f"
    )

    store_and_fwd_option = st.selectbox(
        "Store and Forward Flag",
        options=["N", "Y"]
    )

    store_and_fwd_flag = 0 if store_and_fwd_option == "N" else 1


# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------

haversine_distance_km = haversine_distance(
    pickup_latitude,
    pickup_longitude,
    dropoff_latitude,
    dropoff_longitude
)

manhattan_distance_km = manhattan_distance(
    pickup_latitude,
    pickup_longitude,
    dropoff_latitude,
    dropoff_longitude
)

longitude_difference = (
    dropoff_longitude - pickup_longitude
)

latitude_difference = (
    dropoff_latitude - pickup_latitude
)

bearing = calculate_bearing(
    pickup_latitude,
    pickup_longitude,
    dropoff_latitude,
    dropoff_longitude
)

is_weekend = int(
    pickup_weekday in [5, 6]
)

is_rush_hour = int(
    pickup_hour in [7, 8, 9, 16, 17, 18, 19]
)

is_night = int(
    pickup_hour in [0, 1, 2, 3, 4, 5]
)


# ---------------------------------------------------------
# Build Input DataFrame
# ---------------------------------------------------------

input_data = pd.DataFrame([{
    "vendor_id": vendor_id,
    "passenger_count": passenger_count,
    "pickup_longitude": pickup_longitude,
    "pickup_latitude": pickup_latitude,
    "dropoff_longitude": dropoff_longitude,
    "dropoff_latitude": dropoff_latitude,
    "store_and_fwd_flag": store_and_fwd_flag,
    "pickup_hour": pickup_hour,
    "pickup_day": pickup_day,
    "pickup_month": pickup_month,
    "pickup_weekday": pickup_weekday,
    "is_weekend": is_weekend,
    "is_rush_hour": is_rush_hour,
    "is_night": is_night,
    "haversine_distance_km": haversine_distance_km,
    "manhattan_distance_km": manhattan_distance_km,
    "longitude_difference": longitude_difference,
    "latitude_difference": latitude_difference,
    "bearing": bearing
}])


# ---------------------------------------------------------
# Match Model Feature Names and Order
# ---------------------------------------------------------

expected_features = get_expected_features(model)

missing_features = [
    feature
    for feature in expected_features
    if feature not in input_data.columns
]

extra_features = [
    feature
    for feature in input_data.columns
    if feature not in expected_features
]

if missing_features:
    st.error(
        "The following model features are missing from the app input: "
        + ", ".join(missing_features)
    )
    st.stop()

if extra_features:
    input_data = input_data.drop(
        columns=extra_features
    )

input_data = input_data.reindex(
    columns=expected_features
)


# ---------------------------------------------------------
# Display Engineered Trip Information
# ---------------------------------------------------------

st.subheader("📍 Calculated Trip Features")

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric(
    "Haversine Distance",
    f"{haversine_distance_km:.2f} km"
)

metric_col2.metric(
    "Manhattan Distance",
    f"{manhattan_distance_km:.2f} km"
)

metric_col3.metric(
    "Bearing",
    f"{bearing:.2f}°"
)

with st.expander("View Model Input Features"):
    st.dataframe(
        input_data,
        use_container_width=True
    )

    st.write(
        "Model expects",
        len(expected_features),
        "features."
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button(
    "Predict Trip Duration",
    type="primary",
    use_container_width=True
):
    try:
        prediction_seconds = float(
            model.predict(input_data)[0]
        )

        prediction_seconds = max(
            prediction_seconds,
            0
        )

        prediction_minutes = (
            prediction_seconds / 60
        )

        prediction_hours = (
            prediction_seconds / 3600
        )

        st.success(
            f"Predicted Trip Duration: "
            f"{prediction_seconds:.2f} seconds"
        )

        if prediction_minutes < 60:
            st.info(
                f"Approximate Duration: "
                f"{prediction_minutes:.2f} minutes"
            )
        else:
            st.info(
                f"Approximate Duration: "
                f"{prediction_hours:.2f} hours"
            )

    except ValueError as error:
        st.error(
            f"Feature mismatch while predicting: {error}"
        )

        st.write(
            "Model feature names:",
            expected_features
        )

        st.write(
            "App input feature names:",
            input_data.columns.tolist()
        )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )