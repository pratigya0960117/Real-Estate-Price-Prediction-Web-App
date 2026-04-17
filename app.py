import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🏠 Real Estate Price Prediction")

area = st.number_input("Area (sq ft)")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")

if st.button("Predict Price"):
    features = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(features)

    st.success(f"Estimated Price: ₹ {prediction[0]:.2f} Lakhs")
