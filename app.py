import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("models/model.pkl")

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction")
st.markdown("Predict house prices using machine learning")

st.divider()

# Layout in columns
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Lot Area (sq ft)", value=5000)
    bedrooms = st.number_input("Bedrooms", value=3)
    bathrooms = st.number_input("Bathrooms", value=2)

with col2:
    garage = st.number_input("Garage Capacity", value=1)
    stories = st.selectbox("Stories", [1, 2, 3])
    aircon = st.selectbox("Air Conditioning", ["Yes", "No"])

# Convert input to DataFrame
input_data = pd.DataFrame({
    "Lot Area": [area],
    "Bedroom AbvGr": [bedrooms],
    "Full Bath": [bathrooms],
    "Garage Cars": [garage],
    "House Style_2Story": [1 if stories == 2 else 0],
    "Central Air_Y": [1 if aircon == "Yes" else 0]
})

# Match training columns
model_features = model.feature_names_in_

for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model_features]

st.divider()

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: ${int(prediction):,}")