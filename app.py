import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load Models and Encoders
@st.cache_resource
def load_models():
    models = {}
    encoders = {}
    try:
        models["XGBoost"] = joblib.load("xgboost_model.pkl")
        models["Random Forest"] = joblib.load("random_forest_model.pkl")
        models["Hybrid Model"] = joblib.load("hybrid_model.pkl")
        encoders["scaler"] = joblib.load("scaler.pkl")
        encoders["label_encoders"] = joblib.load("label_encoders.pkl")
        encoders["target_encoder"] = joblib.load("target_encoder.pkl")
    except FileNotFoundError as e:
        st.error(f"⚠ Model or encoder file not found: {e}")
        return None, None
    return models, encoders

models, encoders = load_models()

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ai_job_market_insights.csv")
    df.dropna(inplace=True)
    return df

df = load_data()

st.title("📊 AI-Powered Job Market Insights")
st.write("Explore job trends, salaries, and predict job growth using ML models.")

# Data Preview
if st.checkbox("🔍 Show Dataset"):
    st.write(df.head())

# Visualization - Salary Distribution
st.subheader("💰 Salary Distribution by Industry")
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x="Industry", y="Salary_USD", data=df, palette="coolwarm", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# AI Adoption Across Industries
st.subheader("🚀 AI Adoption Levels by Industry")
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x="Industry", hue="AI_Adoption_Level", data=df, palette="viridis", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# ML Model Predictions
st.subheader("🔮 Job Growth Predictions")
st.write("Select inputs to predict job growth potential:")

# User Inputs
industry = st.selectbox("🏢 Industry", df["Industry"].unique())
company_size = st.selectbox("💰 Company Size", df["Company_Size"].unique())
skill = st.selectbox("🛠 Required Skills", df["Required_Skills"].unique())
salary = st.number_input("💵 Salary (USD)", min_value=30000, max_value=300000, step=5000)

if st.button("🔍 Predict Job Growth"):
    if models and encoders:
        # Prepare input
        input_dict = {
            "Industry": industry,
            "Company_Size": company_size,
            "Required_Skills": skill,
            "Salary_USD": salary
        }
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical inputs
        for col in ["Industry", "Company_Size", "Required_Skills"]:
            input_df[col] = encoders["label_encoders"][col].transform([input_df[col][0]])
        
        # Scale numerical input
        input_df["Salary_USD"] = encoders["scaler"].transform(input_df[["Salary_USD"]])
        
        # Predict
        rf_pred = models["Random Forest"].predict(input_df)[0]
        xgb_pred = models["XGBoost"].predict(input_df)[0]
        hybrid_pred = models["Hybrid Model"].predict(input_df)[0]
        
        # Decode predictions
        rf_pred_decoded = encoders["target_encoder"].inverse_transform([int(rf_pred)])[0]
        xgb_pred_decoded = encoders["target_encoder"].inverse_transform([int(xgb_pred)])[0]
        hybrid_pred_decoded = encoders["target_encoder"].inverse_transform([int(hybrid_pred)])[0]
        
        st.success(f"🌟 Random Forest Prediction: {rf_pred_decoded}")
        st.success(f"🔥 XGBoost Prediction: {xgb_pred_decoded}")
        st.success(f"🔮 Hybrid Model Prediction: {hybrid_pred_decoded}")
    else:
        st.error("⚠ Models or encoders not loaded. Check if files are uploaded.")

st.sidebar.header("About")
st.sidebar.write("Built with Random Forest, XGBoost, and a hybrid model to predict job growth trends in the AI job market.")