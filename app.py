import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from openrouteservice import Client
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import math

# ============================================
# CONFIG
# ============================================

st.set_page_config(page_title="AI Delivery Optimization System", layout="wide")
st.title("🚀 AI-Powered Intelligent Delivery Time Optimization System")

ORS_API_KEY = "YOUR_ORS_KEY"

# ============================================
# LOAD MODELS
# ============================================

preprocessor = joblib.load("preprocessing.pkl")
lr_model = joblib.load("linear_regression.pkl")
dt_model = joblib.load("decision_tree.pkl")
rf_model = joblib.load("random_forest.pkl")

models = {
    "Linear Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model
}

# ============================================
# INPUT PANEL
# ============================================

col1, col2, col3 = st.columns([1,1,1.5])

with col1:
    st.subheader("📦 Delivery Details")

    order_date = st.date_input("Order Date")
    order_time = st.time_input("Order Time")
    pickup_time = st.time_input("Pickup Time")

    multiple_deliveries = st.slider("Multiple Deliveries", 1, 5, 1)

    weather = st.selectbox("Weather", ["Sunny","Cloudy","Rainy","Stormy","Fog"])
    traffic = st.selectbox("Traffic", ["Low","Medium","High","Jam"])
    order_type = st.selectbox("Order Type", ["Meat","Vegetables","Meat or Vegetables"])
    vehicle = st.selectbox("Vehicle", ["Bike","Car","Scooter"])

with col2:
    st.subheader("📍 Location")

    rest_lat = st.number_input("Restaurant Latitude", value=12.9716, format="%.6f")
    rest_lon = st.number_input("Restaurant Longitude", value=77.5946, format="%.6f")
    del_lat = st.number_input("Delivery Latitude", value=12.9352, format="%.6f")
    del_lon = st.number_input("Delivery Longitude", value=77.6245, format="%.6f")

# ============================================
# FEATURE ENGINEERING
# ============================================

Order_Day = order_date.day
Order_Month = order_date.month
Order_Hour = order_time.hour
Pickup_Hour = pickup_time.hour

input_data = {
    "Restaurant_latitude": rest_lat,
    "Restaurant_longitude": rest_lon,
    "Delivery_location_latitude": del_lat,
    "Delivery_location_longitude": del_lon,
    "multiple_deliveries": multiple_deliveries,
    "Order_Day": Order_Day,
    "Order_Month": Order_Month,
    "Order_Hour": Order_Hour,
    "Pickup_Hour": Pickup_Hour,
    "Weatherconditions": weather,
    "Road_traffic_density": traffic,
    "Type_of_order": order_type,
    "Type_of_vehicle": vehicle
}

# ============================================
# PREDICTION
# ============================================

if st.button("🔮 Predict Delivery Time"):

    df_input = pd.DataFrame([input_data])
    processed = preprocessor.transform(df_input)

    predictions = {}
    for name, model in models.items():
        predictions[name] = round(model.predict(processed)[0], 2)

    # ETA calculation using Random Forest as default
    best_model = "Random Forest"
    predicted_minutes = predictions[best_model]
    eta = datetime.combine(order_date, order_time) + timedelta(minutes=predicted_minutes)

    # ========================================
    # DISPLAY PREDICTIONS
    # ========================================

    with col2:
        st.subheader("📊 Model Predictions")
        for name, value in predictions.items():
            st.metric(name, f"{value} min")

        st.success(f"🏆 Recommended Model: {best_model}")
        st.info(f"🕒 Estimated Arrival Time: {eta.strftime('%I:%M %p')}")

    # ========================================
    # MAP
    # ========================================

    with col3:
        st.subheader("🗺️ Route Visualization")
        m = folium.Map(location=[(rest_lat+del_lat)/2, (rest_lon+del_lon)/2], zoom_start=13)
        folium.Marker([rest_lat, rest_lon], tooltip="Restaurant", icon=folium.Icon(color='green')).add_to(m)
        folium.Marker([del_lat, del_lon], tooltip="Delivery", icon=folium.Icon(color='red')).add_to(m)
        folium.PolyLine([(rest_lat,rest_lon),(del_lat,del_lon)], color="blue").add_to(m)
        st_folium(m, width=700, height=500)

    # ========================================
    # FEATURE IMPORTANCE
    # ========================================

    with st.expander("📈 Feature Importance (Random Forest)"):
        importances = rf_model.feature_importances_
        features = preprocessor.feature_names_in_
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances[:len(features)]
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))
