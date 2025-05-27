import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
&nbsp;
&nbsp;

# Load Dataset
df = pd.read_csv("ai_job_market_insights.csv")
&nbsp;
&nbsp;

# Define categorical and numerical columns
categorical_columns = ["Industry", "Company_Size", "Required_Skills"]
numerical_columns = ["Salary_USD"]
&nbsp;
&nbsp;

# Encode categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
&nbsp;
&nbsp;

# Handle missing values
df = df.dropna()
&nbsp;
&nbsp;

# Scale numerical data
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
&nbsp;
&nbsp;

# Define features and target
X = df[categorical_columns + numerical_columns]
y = df["Job_Growth_Projection"].astype(str)
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
&nbsp;
&nbsp;

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
&nbsp;
&nbsp;

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
&nbsp;
&nbsp;

# Train XGBoost Model
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
&nbsp;
&nbsp;

# Train Keras Deep Learning Model
def create_keras_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification for job growth
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
&nbsp;
&nbsp;

keras_model = create_keras_model(X_train.shape[1])
keras_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
&nbsp;
&nbsp;

# Save Models & Encoders
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")
keras_model.save("keras_model.h5")  # Save Keras model in HDF5 format
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
&nbsp;
&nbsp;

print("✅ All models and encoders saved successfully!")