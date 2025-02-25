{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "15682260-0a31-4133-b2fd-5160b75e4326",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\keras\\src\\layers\\core\\dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Models trained & saved successfully!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from xgboost import XGBRegressor\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense\n",
    "\n",
    "# ✅ Load Dataset\n",
    "df = pd.read_csv(\"ai_job_market_insights.csv\")\n",
    "\n",
    "# ✅ Encode Categorical Variables\n",
    "label_encoders = {}\n",
    "categorical_columns = [\"Job_Title\", \"Industry\", \"Company_Size\", \"Location\", \n",
    "                       \"AI_Adoption_Level\", \"Automation_Risk\", \"Required_Skills\", \n",
    "                       \"Remote_Friendly\", \"Job_Growth_Projection\"]\n",
    "\n",
    "for col in categorical_columns:\n",
    "    le = LabelEncoder()\n",
    "    df[col] = le.fit_transform(df[col])  # Convert text to numbers\n",
    "    label_encoders[col] = le  \n",
    "\n",
    "# ✅ Scale Salary\n",
    "scaler = StandardScaler()\n",
    "df[\"Salary_USD\"] = scaler.fit_transform(df[[\"Salary_USD\"]])\n",
    "\n",
    "# ✅ Select Features & Target\n",
    "X = df.drop(columns=[\"Salary_USD\"])  # Features\n",
    "y = df[\"Salary_USD\"]  # Target\n",
    "\n",
    "# ✅ Train/Test Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# ✅ Train Random Forest Model\n",
    "rf_model = RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "rf_model.fit(X_train, y_train)\n",
    "joblib.dump(rf_model, \"random_forest_model.pkl\")\n",
    "\n",
    "# ✅ Train XGBoost Model\n",
    "xgb_model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)\n",
    "xgb_model.fit(X_train, y_train)\n",
    "joblib.dump(xgb_model, \"xgboost_model.pkl\")\n",
    "\n",
    "# ✅ Train Deep Learning Model\n",
    "deep_model = Sequential([\n",
    "    Dense(64, activation=\"relu\", input_shape=(X_train.shape[1],)),\n",
    "    Dense(32, activation=\"relu\"),\n",
    "    Dense(1)\n",
    "])\n",
    "deep_model.compile(optimizer=\"adam\", loss=\"mse\")\n",
    "deep_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)\n",
    "deep_model.save(\"deep_learning_model.keras\")\n",
    "\n",
    "# ✅ Train Hybrid Model (Average of XGBoost & Random Forest)\n",
    "rf_preds = rf_model.predict(X_test)\n",
    "xgb_preds = xgb_model.predict(X_test)\n",
    "hybrid_preds = (rf_preds + xgb_preds) / 2\n",
    "joblib.dump(hybrid_preds, \"hybrid_model.pkl\")\n",
    "\n",
    "print(\"✅ Models trained & saved successfully!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b2be5867-417e-4468-8676-ce3a4d42f478",
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
