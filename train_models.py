import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# ✅ Load Dataset
df = pd.read_csv("ai_job_market_insights.csv")

# 🔹 Define categorical and numerical columns
categorical_columns = ["Job_Title", "Industry", "Company_Size", "Location",
                       "AI_Adoption_Level", "Automation_Risk", "Required_Skills", "Remote_Friendly"]

numerical_columns = ["Salary_USD"]

# ✅ Encode categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ✅ Handle missing values
df = df.dropna()

# ✅ Scale numerical data
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# ✅ Define features and target
X = df.drop(columns=["Job_Growth_Projection"])  # Removing target variable
y = df["Job_Growth_Projection"].astype(str)  # Converting to string before encoding

# ✅ Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ✅ Train XGBoost Model
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# ✅ Save Models & Encoders
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("✅ All models and encoders saved successfully!")
