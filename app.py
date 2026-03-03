# ============================================
# AI-POWERED DELIVERY TIME OPTIMIZATION SYSTEM
# Advanced Thesis Version - CLEAN
# ============================================

import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import math

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(page_title="AI Delivery Optimization System", layout="wide")
st.title("🚀 AI-Powered Intelligent Delivery Time Optimization System")

# ============================================
# LOAD FULL PIPELINE MODELS
# ============================================

@st.cache_resource
def load_models():
    lr = joblib.load("linear_regression_pipeline.pkl")
    dt = joblib.load("decision_tree_pipeline.pkl")
    rf = joblib.load("random_forest_pipeline.pkl")
    return lr, dt, rf

lr_model, dt_model, rf_model = load_models()

models = {
    "Linear Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model
}

# ============================================
# LAYOUT
# ============================================

col_input, col_prediction, col_map = st.columns([1,1,1.4])

# ============================================
# INPUT SECTION
# ============================================

with col_input:
    st.subheader("📦 Delivery Information")

    order_date = st.date_input("Order Date")
    order_time = st.time_input("Order Time")
    pickup_time = st.time_input("Pickup Time")

    multiple_deliveries = st.slider("Multiple Deliveries", 1, 5, 1)
    shelf_life = st.slider("Shelf Life (days)", 1, 10, 3)

    weather = st.selectbox("Weather Conditions",
                           ["Sunny","Cloudy","Rainy","Stormy","Fog"])

    traffic = st.selectbox("Traffic Density",
                           ["Low","Medium","High","Jam"])

    order_type = st.selectbox("Type of Order",
                              ["Meat","Vegetables","Meat or Vegetables"])

    vehicle = st.selectbox("Vehicle Type",
                           ["Bike","Car","Scooter"])

    st.subheader("📍 Location Details")

    rest_lat = st.number_input("Restaurant Latitude", value=12.9716, format="%.6f")
    rest_lon = st.number_input("Restaurant Longitude", value=77.5946, format="%.6f")
    del_lat = st.number_input("Delivery Latitude", value=12.9352, format="%.6f")
    del_lon = st.number_input("Delivery Longitude", value=77.6245, format="%.6f")

# ============================================
# FEATURE ENGINEERING (MATCH TRAINING EXACTLY)
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

# ============================================
# DISTANCE CALCULATOR
# ============================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ============================================
# PREDICTION
# ============================================

if st.button("🔮 Predict Delivery Time"):

    df_input = pd.DataFrame([input_data])

    predictions = {}
    for name, model in models.items():
        predictions[name] = round(model.predict(df_input)[0], 2)

    best_model = "Random Forest"
    predicted_minutes = predictions[best_model]

    eta = datetime.combine(order_date, order_time) + timedelta(minutes=predicted_minutes)

    # ========================================
    # DISPLAY RESULTS
    # ========================================

    with col_prediction:
        st.subheader("📊 Model Predictions")

        for name, value in predictions.items():
            st.metric(name, f"{value} minutes")

        st.success(f"🏆 Recommended Model: {best_model}")
        st.info(f"🕒 Estimated Arrival Time: {eta.strftime('%I:%M %p')}")

        straight_distance = haversine(rest_lat, rest_lon, del_lat, del_lon)
        st.markdown(f"**📏 Straight-Line Distance:** {straight_distance:.2f} km")

    # ========================================
    # MAP VISUALIZATION
    # ========================================

    with col_map:
        st.subheader("🗺️ Route Visualization")

        m = folium.Map(location=[(rest_lat+del_lat)/2,
                                 (rest_lon+del_lon)/2], zoom_start=13)

        folium.Marker([rest_lat, rest_lon],
                      tooltip="Restaurant",
                      icon=folium.Icon(color="green")).add_to(m)

        folium.Marker([del_lat, del_lon],
                      tooltip="Delivery",
                      icon=folium.Icon(color="red")).add_to(m)

        folium.PolyLine([(rest_lat,rest_lon),
                         (del_lat,del_lon)],
                        color="blue").add_to(m)

        st_folium(m, width=700, height=500)

    # ========================================
    # FEATURE IMPORTANCE
    # ========================================

    with st.expander("📈 Feature Importance (Random Forest)"):
        try:
            rf_step = rf_model.named_steps["model"]
            preprocessor_step = rf_model.named_steps["preprocessing"]

            feature_names = preprocessor_step.get_feature_names_out()
            importances = rf_step.feature_importances_

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature").head(10))

        except:
            st.warning("Feature importance unavailable.")
