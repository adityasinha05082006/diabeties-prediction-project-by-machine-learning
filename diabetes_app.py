import streamlit as st
import numpy as np
import joblib

st.title("Diabetic Prediction App")

# Load model
model = joblib.load(r"C:\Users\Aditya Singh\Desktop\Diabetic prediction\diabetes_model.pkl")

# Input features
preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("BloodPressure", min_value=0)
skin = st.number_input("SkinThickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0)
age = st.number_input("Age", min_value=0)

features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("The person is likely Diabetic.")
    else:
        st.success("The person is NOT Diabetic.")
        
        

