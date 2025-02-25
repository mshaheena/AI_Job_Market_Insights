{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6480ab0d-0a4e-4d6d-9219-fd6f5ca77e10",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Random Forest Model Retrained & Saved Successfully!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import joblib\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "\n",
    "# ✅ Load dataset\n",
    "df = pd.read_csv(\"ai_job_market_insights.csv\")\n",
    "\n",
    "# ✅ Identify Categorical Columns\n",
    "categorical_cols = [\"Job_Title\", \"Industry\", \"Company_Size\", \"Location\", \n",
    "                    \"AI_Adoption_Level\", \"Automation_Risk\", \"Required_Skills\", \n",
    "                    \"Remote_Friendly\", \"Job_Growth_Projection\"]  # 🔹 Added Job_Growth_Projection\n",
    "\n",
    "# ✅ Convert Categorical Features to Numeric (Label Encoding)\n",
    "label_encoders = {}\n",
    "for col in categorical_cols:\n",
    "    le = LabelEncoder()\n",
    "    df[col] = le.fit_transform(df[col])\n",
    "    label_encoders[col] = le  # Store encoders for later use\n",
    "\n",
    "# ✅ Define Features (X) & Target (y)\n",
    "X = df.drop(columns=[\"Salary_USD\"])  # Features\n",
    "y = df[\"Salary_USD\"]  # Target Variable\n",
    "\n",
    "# ✅ Scale Numerical Features\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# ✅ Train Random Forest Model\n",
    "rf_model = RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "rf_model.fit(X_scaled, y)\n",
    "\n",
    "# ✅ Save Model & Scaler\n",
    "joblib.dump(rf_model, \"random_forest_model.pkl\")\n",
    "joblib.dump(scaler, \"scaler.pkl\")  # Save scaler to preprocess new data\n",
    "\n",
    "print(\"✅ Random Forest Model Retrained & Saved Successfully!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b63b41ae-6621-407d-acbd-f2fadb7dc3c8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Scaler saved successfully as 'scaler.pkl'!\n"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "import pandas as pd\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# 📌 Load dataset\n",
    "df = pd.read_csv(\"C:/Users/Dell/Desktop/AI_Job_Market_Insights/ai_job_market_insights.csv\")\n",
    "\n",
    "# 📌 Select only numerical columns for scaling\n",
    "numerical_columns = [\"Salary_USD\"]  # Modify if needed\n",
    "df_numeric = df[numerical_columns]\n",
    "\n",
    "# 📌 Fit the scaler\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(df_numeric)  # Fit on numerical columns\n",
    "\n",
    "# 📌 Save the scaler\n",
    "joblib.dump(scaler, \"C:/Users/Dell/Desktop/AI_Job_Market_Insights/scaler.pkl\")\n",
    "\n",
    "print(\"✅ Scaler saved successfully as 'scaler.pkl'!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "472059af-a4fe-47f3-9652-c4d7565992d9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
