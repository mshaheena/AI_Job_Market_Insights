import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load Models with Exception Handling
@st.cache_resource
def load_models():
    models = {}
    try:
        models["XGBoost"] = joblib.load("xgboost_model.pkl")
    except:
        st.warning("⚠ XGBoost Model Not Found!")
    
    try:
        models["Random Forest"] = joblib.load("random_forest_model.pkl")
    except:
        st.warning("⚠ Random Forest Model Not Found!")
    
    try:
        models["Hybrid Model"] = joblib.load("hybrid_model.pkl")
    except:
        st.warning("⚠ Hybrid Model Not Found!")
    
    return models

models = load_models()

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ai_job_market_insights.csv")
    df.dropna(inplace=True)
    return df

df = load_data()

st.title("📊 AI-Powered Job Market Insights")
st.write("Explore job trends, salaries, and AI adoption levels using ML models.")

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
st.subheader("🔮 AI Job Market Predictions")
st.write("Select inputs to predict AI Skill Rankings:")

# User Inputs
industry = st.selectbox("🏢 Industry", df["Industry"].unique())
income_group = st.selectbox("💰 Company Size", df["Company_Size"].unique())
skill = st.selectbox("🛠 Required Skills", df["Required_Skills"].unique())
salary = st.number_input("💵 Salary (USD)", min_value=30000, max_value=300000, step=5000)

if st.button("🔍 Predict AI Job Market Insights"):
    input_features = np.array([[industry, income_group, skill, salary]])
    input_features = input_features.astype(float)
    
    # Predictions
    try:
        rf_prediction = models["Random Forest"].predict(input_features)[0]
        st.success(f"🌟 Random Forest Prediction: {rf_prediction:.2f}")
    except:
        st.error("⚠ Random Forest Model Not Loaded")
    
    try:
        xgb_prediction = models["XGBoost"].predict(input_features)[0]
        st.success(f"🔥 XGBoost Prediction: {xgb_prediction:.2f}")
    except:
        st.error("⚠ XGBoost Model Not Loaded")
    
    try:
        hybrid_prediction = models["Hybrid Model"].predict(input_features)[0]
        st.success(f"🔮 Hybrid Model Prediction: {hybrid_prediction:.2f}")
    except:
        st.error("⚠ Hybrid Model Not Loaded")
