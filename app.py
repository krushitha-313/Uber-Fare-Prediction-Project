import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math

# Load model
with open("fare_model.pkl", "rb") as f:
    model = pickle.load(f)

# Haversine Distance Function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# Streamlit UI
st.title("Uber Fare Prediction 🚖")

pickup_lat = st.number_input("Pickup Latitude")
pickup_lon = st.number_input("Pickup Longitude")
dropoff_lat = st.number_input("Dropoff Latitude")
dropoff_lon = st.number_input("Dropoff Longitude")
passengers = st.number_input("Number of Passengers", min_value=1, max_value=10, step=1)

if st.button("Predict Fare"):
    distance = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    features = np.array([[pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, passengers, distance]])
    fare = model.predict(features)[0]
    st.success(f"Estimated Fare: ${fare:.2f}")
