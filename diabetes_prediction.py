import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Data
df = pd.read_csv(r"C:\Users\Aditya Singh\Desktop\Diabetic prediction\diabetes.csv")

# 2. Data Preprocessing
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. Save Model
joblib.dump(model, "diabetes_model.pkl")


