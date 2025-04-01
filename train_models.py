import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Load Dataset
df = pd.read_csv("ai_job_market_insights.csv")

# Define categorical and numerical columns
categorical_columns = ["Industry", "Company_Size", "Required_Skills"]  # Match app.py inputs
numerical_columns = ["Salary_USD"]

# Encode categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Handle missing values
df = df.dropna()

# Scale numerical data
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Define features and target (using a subset for simplicity)
X = df[categorical_columns + numerical_columns]
y = df["Job_Growth_Projection"].astype(str)  # Assuming this is the target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost Model
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Simple Hybrid Model (average predictions)
class HybridModel:
    def __init__(self, rf, xgb):
        self.rf = rf
        self.xgb = xgb
    def predict(self, X):
        rf_preds = self.rf.predict(X)
        xgb_preds = self.xgb.predict(X)
        return (rf_preds + xgb_preds) / 2

hybrid_model = HybridModel(rf_model, xgb_model)

# Save Models & Encoders
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")
joblib.dump(hybrid_model, "hybrid_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("✅ All models and encoders saved successfully!")