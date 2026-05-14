
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('house_price_model.pkl')

# Title
st.title("House Price Predictor")

# Input fields
lot_area = st.number_input("Lot Area")
overall_qual = st.number_input("Overall Quality")
overall_cond = st.number_input("Overall Condition")
year_built = st.number_input("Year Built")
total_bsmt_sf = st.number_input("Basement Area")
gr_liv_area = st.number_input("Living Area")
full_bath = st.number_input("Full Bathrooms")
bedroom = st.number_input("Bedrooms")
garage_cars = st.number_input("Garage Cars")

# Prediction button
if st.button("Predict"):

    features = np.array([[
        lot_area,
        overall_qual,
        overall_cond,
        year_built,
        total_bsmt_sf,
        gr_liv_area,
        full_bath,
        bedroom,
        garage_cars
    ]])

    prediction = model.predict(features)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
