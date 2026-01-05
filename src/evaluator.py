# src/evaluator.py

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error # <--- ADD THIS HERE!
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_feature_importance(model, X_train):
    """Calculates and returns the top 10 most important features."""
    
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top_10_features = feature_importances.nlargest(10)
    
    print("\n[Evaluator] Top 10 Feature Importance:")
    print(top_10_features)
    
    return top_10_features

def print_performance_metrics(Y_test, Y_pred):
    """Calculates and prints the final performance metrics."""
    
    mae = mean_absolute_error(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)
    
    print("\n--- Final Model Performance Metrics ---")
    print(f"R-squared Score (R2): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:,.2f} hg/ha")
    
    return r2, mae