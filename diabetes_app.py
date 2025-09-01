import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('diabetes_model.pkl')

# Title and description
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on medical parameters.
Please enter the required information below and click **Predict** to get the result.
""")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This prediction is based on a machine learning model trained on medical data.
    **Important:** This is not a substitute for professional medical advice.
    """)

    st.header("📊 Sample Inputs")
    st.markdown("""
    **Non-Diabetic Sample:**
    - Pregnancies: 1
    - Glucose: 85
    - Blood Pressure: 66
    - Skin Thickness: 29
    - Insulin: 0
    - BMI: 26.6
    - Diabetes Pedigree Function: 0.351
    - Age: 31

    **Diabetic Sample:**
    - Pregnancies: 6
    - Glucose: 148
    - Blood Pressure: 72
    - Skin Thickness: 35
    - Insulin: 0
    - BMI: 33.6
    - Diabetes Pedigree Function: 0.627
    - Age: 50
    """)

# Input fields with improved UI
st.header("📝 Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=0,
        help="Number of times pregnant (0-20)"
    )

    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=0,
        max_value=200,
        value=100,
        help="Plasma glucose concentration (0-200 mg/dL)"
    )

    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=0,
        max_value=140,
        value=70,
        help="Diastolic blood pressure (0-140 mm Hg)"
    )

    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=20,
        help="Triceps skin fold thickness (0-100 mm)"
    )

with col2:
    insulin = st.number_input(
        "Insulin Level (mu U/ml)",
        min_value=0,
        max_value=900,
        value=80,
        help="2-Hour serum insulin (0-900 mu U/ml)"
    )

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1,
        help="Body mass index (0.0-70.0)"
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.001,
        help="Diabetes pedigree function (0.0-3.0)"
    )

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=30,
        help="Age in years (1-120)"
    )

# Predict button
if st.button("🔮 Predict Diabetes Risk", type="primary"):
    # Enhanced input validation
    errors = []

    if glucose == 0:
        errors.append("Glucose level cannot be zero")
    elif glucose < 70 or glucose > 200:
        errors.append("Glucose level should be between 70-200 mg/dL")

    if blood_pressure == 0:
        errors.append("Blood pressure cannot be zero")
    elif blood_pressure < 50 or blood_pressure > 140:
        errors.append("Blood pressure should be between 50-140 mm Hg")

    if bmi == 0.0:
        errors.append("BMI cannot be zero")
    elif bmi < 15.0 or bmi > 60.0:
        errors.append("BMI should be between 15.0-60.0")

    if age < 1 or age > 120:
        errors.append("Age should be between 1-120 years")

    if pregnancies < 0 or glucose < 0 or blood_pressure < 0 or skin_thickness < 0 or insulin < 0 or bmi < 0 or dpf < 0:
        errors.append("All values must be non-negative")

    if errors:
        for error in errors:
            st.error(f"❌ {error}")
    else:
        # Prepare input data
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])

        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0]

        result = 'High Risk (Diabetic)' if prediction[0] == 1 else 'Low Risk (Non-Diabetic)'
        confidence = probability[int(prediction[0])] * 100

        # Display result with enhanced styling
        if prediction[0] == 1:
            st.error(f"⚠️ **Prediction: {result}**")
            st.markdown(f"**Confidence:** {confidence:.1f}%")
            st.warning("""
            **⚠️ Important Notice:**
            This indicates a high risk of diabetes. Please consult a healthcare professional
            for proper diagnosis and treatment. Regular monitoring is recommended.
            """)
        else:
            st.success(f"✅ **Prediction: {result}**")
            st.markdown(f"**Confidence:** {confidence:.1f}%")
            st.info("""
            **ℹ️ Note:**
            This indicates a low risk of diabetes. However, maintaining a healthy lifestyle
            and regular health check-ups are still recommended.
            """)

        # Show input summary
        st.subheader("📊 Input Summary")
        st.markdown("**Patient Information Entered:**")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Pregnancies:** {pregnancies}")
            st.write(f"**Glucose:** {glucose} mg/dL")
            st.write(f"**Blood Pressure:** {blood_pressure} mm Hg")
            st.write(f"**Skin Thickness:** {skin_thickness} mm")

        with col2:
            st.write(f"**Insulin:** {insulin} mu U/ml")
            st.write(f"**BMI:** {bmi:.1f}")
            st.write(f"**Diabetes Pedigree Function:** {dpf:.3f}")
            st.write(f"**Age:** {age} years")
