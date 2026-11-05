# Placement Predictor — a Streamlit web app around our saved model.
# Run locally:   streamlit run app.py

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# model.pkl sits next to this file (works even if the app is run from another folder)
MODEL_PATH = Path(__file__).parent / "model.pkl"


# Load the model once and reuse it (Streamlit re-runs this file on every click)
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

# ---------- Page ----------
st.set_page_config(page_title="Placement Predictor", page_icon="🎓")
st.title("🎓 Student Placement Predictor")
st.write("Enter a student's **CGPA** and **IQ** to predict whether they will get placed.")

# ---------- Inputs ----------
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
iq = st.number_input("IQ", min_value=0, max_value=250, value=120, step=1)

# ---------- Prediction ----------
if st.button("Predict"):
    # Same column names as training data
    student = pd.DataFrame({"cgpa": [cgpa], "iq": [iq]})

    prediction = model.predict(student)[0]
    probability = model.predict_proba(student)[0][1]   # chance of being placed

    if prediction == 1:
        st.success(f"Likely to be PLACED  (probability {probability:.0%})")
    else:
        st.error(f"Likely NOT placed  (probability of placement {probability:.0%})")

    st.progress(float(probability))

st.caption("Model: Logistic Regression trained on 100 students (cgpa, iq → placement).")
