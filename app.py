import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.title("📊 AI-Powered Job Market Insights")
st.write("Explore job trends, salaries, and predict job growth using ML models.")

# Load Models and Encoders with Debugging
@st.cache_resource
def load_models():
    models = {}
    encoders = {}
    files = {
        "XGBoost": "xgboost_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "Hybrid Model": "hybrid_model.pkl",
        "scaler": "scaler.pkl",
        "label_encoders": "label_encoders.pkl",
        "target_encoder": "target_encoder.pkl"
    }
    for name, file in files.items():
        if not os.path.exists(file):
            st.error(f"⚠ File not found: {file}")
            return None, None
        try:
            if name in ["scaler", "label_encoders", "target_encoder"]:
                encoders[name] = joblib.load(file)
            else:
                models[name] = joblib.load(file)
        except Exception as e:
            st.error(f"⚠ Error loading {file}: {e}")
            return None, None
    return models, encoders

models, encoders = load_models()

# Load Dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("ai_job_market_insights.csv")
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        st.error("⚠ Dataset 'ai_job_market_insights.csv' not found!")
        return None
    except Exception as e:
        st.error(f"⚠ Error loading dataset: {e}")
        return None

df = load_data()

# Data Preview
if df is not None and st.checkbox("🔍 Show Dataset"):
    st.write(df.head())

# Visualization - Salary Distribution
if df is not None:
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
industry = st.selectbox("🏢 Industry", df["Industry"].unique() if df is not None else ["N/A"])
company_size = st.selectbox("💰 Company Size", df["Company_Size"].unique() if df is not None else ["N/A"])
skill = st.selectbox("🛠 Required Skills", df["Required_Skills"].unique() if df is not None else ["N/A"])
salary = st.number_input("💵 Salary (USD)", min_value=30000, max_value=300000, step=5000)

if st.button("🔍 Predict Job Growth"):
    if models and encoders and df is not None:
        # Prepare input
        input_dict = {
            "Industry": industry,
            "Company_Size": company_size,
            "Required_Skills": skill,
            "Salary_USD": salary
        }
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical inputs
        try:
            for col in ["Industry", "Company_Size", "Required_Skills"]:
                input_df[col] = encoders["label_encoders"][col].transform([input_df[col][0]])
        except Exception as e:
            st.error(f"⚠ Encoding error: {e}")
            st.stop()
        
        # Scale numerical input
        try:
            input_df["Salary_USD"] = encoders["scaler"].transform(input_df[["Salary_USD"]])
        except Exception as e:
            st.error(f"⚠ Scaling error: {e}")
            st.stop()
        
        # Predict
        try:
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
        except Exception as e:
            st.error(f"⚠ Prediction error: {e}")
    else:
        st.error("⚠ Models, encoders, or dataset not loaded. Check uploaded files.")

st.sidebar.header("About")
st.sidebar.write("Built with Random Forest, XGBoost, and a hybrid model to predict job growth trends in the AI job market.")