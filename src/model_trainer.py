# src/model_trainer.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import numpy as np
import os

# Define the path where the trained model will be saved
MODEL_SAVE_PATH = 'models/random_forest_model.joblib'

def train_and_save_model(X_train, X_test, Y_train, Y_test):
    """Trains the Random Forest model and saves it."""
    
    # 1. Select the Model: Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    print("\n[Model Trainer] Starting Random Forest training...")

    # 2. Train the Model
    rf_model.fit(X_train, Y_train)
    print("   Training complete!")

    # 3. Evaluate (Quick Check)
    Y_pred = rf_model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    
    print(f"   Test R-squared Score: {r2:.4f}")
    print(f"   Test MAE: {mae:,.2f} hg/ha")

    # 4. Save the Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, MODEL_SAVE_PATH)
    print(f"[Model Trainer] Model saved successfully to {MODEL_SAVE_PATH}")
    
    return rf_model, Y_pred