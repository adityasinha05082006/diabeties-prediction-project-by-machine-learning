import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open(r"C:\Users\Aditya Singh\OneDrive\Desktop\Diabetic prediction\diabetes_model.pkl", 'rb'))

# Title
st.title("Diabetes Prediction App")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
blood_pressure = st.number_input("Blood Pressure", 0, 140)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
    st.success(f"The person is {result}")
