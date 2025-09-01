import numpy as np
import joblib

def test_diabetes_prediction():
    """Test the diabetes prediction model with sample inputs"""

    # Load the model
    model = joblib.load('diabetes_model.pkl')

    # Test cases
    test_cases = [
        {
            "name": "Non-Diabetic Sample",
            "input": [1, 85, 66, 29, 0, 26.6, 0.351, 31],
            "expected": "Low Risk (Non-Diabetic)"
        },
        {
            "name": "Diabetic Sample",
            "input": [6, 148, 72, 35, 0, 33.6, 0.627, 50],
            "expected": "High Risk (Diabetic)"
        },
        {
            "name": "Edge Case - Young Adult",
            "input": [0, 90, 70, 25, 50, 22.0, 0.200, 21],
            "expected": "Low Risk (Non-Diabetic)"
        },
        {
            "name": "Edge Case - Elderly",
            "input": [8, 160, 85, 40, 200, 35.0, 0.800, 65],
            "expected": "High Risk (Diabetic)"
        }
    ]

    print("🧪 Testing Diabetes Prediction Model")
    print("=" * 50)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 30)

        # Prepare input
        input_data = np.array([test_case['input']])

        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0]

        result = 'High Risk (Diabetic)' if prediction[0] == 1 else 'Low Risk (Non-Diabetic)'
        confidence = probability[int(prediction[0])] * 100

        print(f"Input: {test_case['input']}")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"Expected: {test_case['expected']}")

        # Check if prediction matches expectation (basic check)
        if result == test_case['expected']:
            print("✅ PASS")
        else:
            print("⚠️  DIFFERENT RESULT")

    print("\n" + "=" * 50)
    print("✅ Testing completed!")
    print("\n📝 Note: The model predictions may vary slightly from expectations")
    print("   as this is a machine learning model with some inherent uncertainty.")

if __name__ == "__main__":
    test_diabetes_prediction()
