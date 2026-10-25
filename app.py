import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="centered")
st.title("🌾 Crop Yield Prediction")
st.write(
    "Estimate crop yield (hg/ha) using the trained machine-learning model from Assignment 3. "
    "This is an educational prediction, not an agricultural guarantee."
)

# These names and their order must match the columns used to train crop_yield.pkl.
FEATURE_COLUMNS = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "log_rainfall",
    "log_pesticides",
    "temp_x_rainfall",
    "temp_squared",
    "crop_Cassava",
    "crop_Maize",
    "crop_Plantains and others",
    "crop_Potatoes",
    "crop_Rice, paddy",
    "crop_Sorghum",
    "crop_Soybeans",
    "crop_Sweet potatoes",
    "crop_Wheat",
    "crop_Yams",
]

CROP_FEATURES = {
    "Cassava": "crop_Cassava",
    "Maize": "crop_Maize",
    "Plantains and others": "crop_Plantains and others",
    "Potatoes": "crop_Potatoes",
    "Rice, paddy": "crop_Rice, paddy",
    "Sorghum": "crop_Sorghum",
    "Soybeans": "crop_Soybeans",
    "Sweet potatoes": "crop_Sweet potatoes",
    "Wheat": "crop_Wheat",
    "Yams": "crop_Yams",
    # All-zero option represents the reference/baseline crop category, if one was
    # omitted when the dataset was one-hot encoded.
    "Other / reference category": None,
}

@st.cache_resource
def load_model():
    with open("crop_yield.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.subheader("Enter crop and environmental information")
crop = st.selectbox("Crop type", list(CROP_FEATURES.keys()))
year = st.number_input("Year", min_value=1900, max_value=2100, value=2020, step=1)
rainfall = st.number_input(
    "Average annual rainfall (mm/year)",
    min_value=0.01, max_value=100000.0, value=1000.0, step=10.0
)
pesticides = st.number_input(
    "Pesticides used (tonnes)",
    min_value=0.01, max_value=10000000.0, value=1000.0, step=10.0
)
avg_temp = st.number_input(
    "Average temperature (°C)",
    min_value=-20.0, max_value=60.0, value=20.0, step=0.1
)

if st.button("Predict crop yield", type="primary"):
    # The notebook's engineered features use natural logarithms and temperature interactions.
    row = {
        "Year": int(year),
        "average_rain_fall_mm_per_year": float(rainfall),
        "pesticides_tonnes": float(pesticides),
        "avg_temp": float(avg_temp),
        "log_rainfall": float(np.log(rainfall)),
        "log_pesticides": float(np.log(pesticides)),
        "temp_x_rainfall": float(avg_temp * np.log(rainfall)),
        "temp_squared": float(avg_temp ** 2),
    }

    selected_feature = CROP_FEATURES[crop]
    for feature in list(CROP_FEATURES.values()):
        if feature is not None:
            row[feature] = int(feature == selected_feature)

    input_data = pd.DataFrame([[row[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    try:
        prediction = float(model.predict(input_data)[0])
        st.success(f"Predicted yield: {prediction:,.2f} hg/ha")
        st.caption("hg/ha means hectograms per hectare. Predictions depend on the dataset and model quality.")
        with st.expander("Show model input features"):
            st.dataframe(input_data, use_container_width=True)
    except Exception as error:
        st.error("Prediction failed. Check that crop_yield.pkl matches the Assignment 3 notebook and installed library versions.")
        st.exception(error)
