import streamlit as st
import numpy as np
import pickle

# Configure the website
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names


model, scaler, feature_names = load_model()

# Website heading
st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the health measurements below to obtain "
    "an experimental machine learning prediction."
)

st.info(
    "This project is for educational purposes only. "
    "It cannot diagnose diabetes or replace professional medical advice."
)

# Input form
with st.form("diabetes_form"):

    st.subheader("Enter Your Health Measurements")

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

    insulin = st.number_input(
        "Insulin Level",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25
    )

    submitted = st.form_submit_button(
        "Predict"
    )

# Make the prediction
if submitted:

    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }

    # Ensure features are in the training order
    input_array = np.array(
        [[input_data[name] for name in feature_names]],
        dtype=float
    )

    # Apply the saved scaler
    scaled_input = scaler.transform(input_array)

    # Generate prediction
    prediction = model.predict(scaled_input)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning(
            "The model predicts a positive diabetes outcome. "
            "This is not a diagnosis. Consult a qualified "
            "healthcare professional for proper evaluation."
        )
    else:
        st.success(
            "The model predicts a negative diabetes outcome. "
            "This does not rule out diabetes or guarantee "
            "that you are free from the condition."
        )