# =====================================================
# AI DELIVERY TIME PREDICTION SYSTEM
# Clean & User-Friendly Version
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import math

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Delivery Time Prediction", layout="wide")
st.title("🚚 AI Delivery Time Prediction System")
st.markdown("Predict estimated delivery time using machine learning models.")

# =====================================================
# LOAD MODELS (FULL PIPELINES)
# =====================================================

@st.cache_resource
def load_models():
    return {
        "Linear Regression": joblib.load("linear_regression_pipeline.pkl"),
        "Decision Tree": joblib.load("decision_tree_pipeline.pkl"),
        "Random Forest": joblib.load("random_forest_pipeline.pkl")
    }

models = load_models()

# =====================================================
# SECTION 1 — DELIVERY DETAILS
# =====================================================

st.header("📦 Delivery Details")

col1, col2 = st.columns(2)

with col1:
    order_date = st.date_input("Order Date")
    order_time = st.time_input("Order Time")
    pickup_time = st.time_input("Pickup Time")
    multiple_deliveries = st.slider("Number of Deliveries", 1, 5, 1)
    shelf_life = st.slider("Shelf Life (days)", 1, 10, 3)

with col2:
    weather = st.selectbox("Weather Condition",
                           ["Sunny","Cloudy","Rainy","Stormy","Fog"])
    traffic = st.selectbox("Traffic Level",
                           ["Low","Medium","High","Jam"])
    order_type = st.selectbox("Order Type",
                              ["Cabbage","Lettuce","Potatoes","Carrots"])
    vehicle = st.selectbox("Vehicle Type",
                           ["Bike","Car","Scooter"])

# =====================================================
# SECTION 2 — LOCATION
# =====================================================

st.header("📍 Location Information")

col3, col4 = st.columns(2)

with col3:
    rest_lat = st.number_input("Restaurant Latitude", value=12.9716, format="%.6f")
    rest_lon = st.number_input("Restaurant Longitude", value=77.5946, format="%.6f")

with col4:
    del_lat = st.number_input("Delivery Latitude", value=12.9352, format="%.6f")
    del_lon = st.number_input("Delivery Longitude", value=77.6245, format="%.6f")

# =====================================================
# FEATURE ENGINEERING
# =====================================================

Order_Day = order_date.day
Order_Month = order_date.month
Order_Hour = order_time.hour
Pickup_Hour = pickup_time.hour

input_data = {
    "Restaurant_latitude": rest_lat,
    "Restaurant_longitude": rest_lon,
    "Delivery_location_latitude": del_lat,
    "Delivery_location_longitude": del_lon,
    "Weatherconditions": weather,
    "Road_traffic_density": traffic,
    "Type_of_order": order_type,
    "Type_of_vehicle": vehicle,
    "multiple_deliveries": multiple_deliveries,
    "Shelf_life(days)": shelf_life,
    "Order_Day": Order_Day,
    "Order_Month": Order_Month,
    "Order_Hour": Order_Hour,
    "Pickup_Hour": Pickup_Hour
}

# =====================================================
# DISTANCE CALCULATION
# =====================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# =====================================================
# PREDICTION BUTTON
# =====================================================

st.markdown("---")

if st.button("🔮 Predict Delivery Time"):

    df_input = pd.DataFrame([input_data])

    predictions = {
        name: round(model.predict(df_input)[0], 2)
        for name, model in models.items()
    }

    best_model = "Random Forest"
    predicted_minutes = predictions[best_model]

    eta = datetime.combine(order_date, order_time) + timedelta(minutes=predicted_minutes)

    # =================================================
    # RESULTS SECTION
    # =================================================

    st.header("📊 Prediction Results")

    colA, colB, colC = st.columns(3)

    colA.metric("Linear Regression", f"{predictions['Linear Regression']} min")
    colB.metric("Decision Tree", f"{predictions['Decision Tree']} min")
    colC.metric("Random Forest", f"{predictions['Random Forest']} min")

    st.success(f"🏆 Recommended Model: {best_model}")
    st.info(f"🕒 Estimated Arrival Time: {eta.strftime('%I:%M %p')}")

    # Distance
    distance = haversine(rest_lat, rest_lon, del_lat, del_lon)
    st.write(f"📏 Straight-Line Distance: {distance:.2f} km")

    # =================================================
    # MAP
    # =================================================

    st.header("🗺️ Route Map")

    m = folium.Map(
        location=[(rest_lat+del_lat)/2, (rest_lon+del_lon)/2],
        zoom_start=13
    )

    folium.Marker(
        [rest_lat, rest_lon],
        tooltip="Restaurant",
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        [del_lat, del_lon],
        tooltip="Delivery Location",
        icon=folium.Icon(color="red")
    ).add_to(m)

    folium.PolyLine(
        [(rest_lat, rest_lon), (del_lat, del_lon)],
        color="blue"
    ).add_to(m)

    st_folium(m, width=900, height=500)
