import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
# ... other imports ...
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Make sure THIS line is present:
from sklearn.model_selection import train_test_split

# IMPORTANT: Check that your file path is correct.
# Assuming your files are inside a folder named 'archive (5)'
data_path = 'archive (5)/' 

# Load the main yield file
try:
    df_yield = pd.read_csv(data_path + 'yield_df.csv')
    print("--- Loaded yield_df.csv ---")
    
    # 1. Print the column names of the main yield file
    print("\nColumn Names in yield_df.csv:")
    print(df_yield.columns.tolist())
    
    # 2. Print the first few rows to see the data
    print("\nFirst 5 Rows of yield_df.csv:")
    print(df_yield.head())
    
except Exception as e:
    print(f"Error loading yield_df.csv: {e}")

# Assuming your loaded DataFrame is named df_yield from the previous step.
df = df_yield.copy()

# 1. Drop the redundant 'Unnamed: 0' column
df = df.drop('Unnamed: 0', axis=1)

# 2. Renaming the yield column for better readability
df = df.rename(columns={'hg/ha_yield': 'Yield'})

print("Cleanup complete. DataFrame head:")
print(df.head())


# Convert categorical columns ('Area' and 'Item') into numerical features
# 'drop_first=True' helps avoid multicollinearity.
df_encoded = pd.get_dummies(df, columns=['Area', 'Item'], drop_first=True)

print("\nData after One-Hot Encoding. New shape:", df_encoded.shape)
print("First 5 rows of encoded data:")
print(df_encoded.head())

# Check and Handle Missing Values (Crucial final cleaning step)
print("\nMissing values check (should be zero or few):")
print(df_encoded.isnull().sum().sort_values(ascending=False).head())

# Drop rows with any remaining missing values for a quick MVP
df_final = df_encoded.dropna()
print(f"\nRemaining rows after dropping NaN: {len(df_final)}")



# 1. Define the Target Variable (Y) and Features (X)
Y = df_final['Yield'] # The yield column
X = df_final.drop('Yield', axis=1) # All other columns are features

# 2. Split data into training (80%) and testing (20%) sets
# random_state=42 ensures your results are reproducible
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("\n--- Phase 2 Complete: Data Split ---")
print(f"X_train shape: {X_train.shape} (Data for training)")
print(f"X_test shape: {X_test.shape} (Data for testing)")


# 1. Select the Model: Random Forest Regressor
# n_estimators=100 is a robust setting for the number of decision trees.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) 

print("Random Forest Regressor initialized.")

# 2. Train the Model: Fit the model to the training data
print("Starting model training... This may take a moment.")
rf_model.fit(X_train, Y_train) 
print("Model training complete!")



# 3. Make Predictions on the test data (X_test)
Y_pred = rf_model.predict(X_test)

print("Predictions generated for the test set (Y_pred).")

# 4. Calculate the Simple Baseline (Average Yield)
# This is what a non-ML model would predict—just the average.
baseline_pred = np.mean(Y_train)
Y_baseline = np.full(Y_test.shape, baseline_pred) 

print(f"Simple Baseline Yield (Average from Training Data): {baseline_pred:.2f} hg/ha")

#phase 4

# 1. Calculate Metrics for our Random Forest Model
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)

print("\n--- Random Forest Model Performance ---")
print(f"Mean Absolute Error (MAE): {mae:,.2f} hg/ha")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} hg/ha")
print(f"R-squared Score (R2): {r2:.4f}")

# 2. Calculate Metrics for the Simple Baseline
baseline_mae = mean_absolute_error(Y_test, Y_baseline)
baseline_r2 = r2_score(Y_test, Y_baseline)

print("\n--- Simple Baseline Performance (Average Guess) ---")
print(f"Baseline MAE: {baseline_mae:,.2f} hg/ha")
print(f"Baseline R2: {baseline_r2:.4f}")

# Analysis: Your R2 score for the Random Forest should be much higher (closer to 1.0) 
# than the Baseline R2 (which will be close to 0), indicating a good model!


# Extract Feature Importance from the trained Random Forest model
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)

# Select the top 10 features
top_10_features = feature_importances.nlargest(10)

print("\n--- Top 10 Most Important Features for Yield Prediction ---")
print(top_10_features)

# Visualize the top 10 features
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_features.values, y=top_10_features.index, palette="viridis")
plt.title('Feature Importance for Crop Yield Prediction (Solving the Black Box)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

# --- Rerunning the visualization (After importing seaborn) ---
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_features.values, y=top_10_features.index, palette="viridis")
plt.title('Feature Importance for Crop Yield Prediction (Solving the Black Box)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show() 

# --- Final Visualization: Actual vs. Predicted Scatter Plot ---
plt.figure(figsize=(8, 8))
plt.scatter(Y_test, Y_pred, alpha=0.6)
# Plot the ideal prediction line (Y=X)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual Yield (hg/ha)')
plt.ylabel('Predicted Yield (hg/ha)')
plt.title('Actual Yield vs. Predicted Yield')
plt.legend()
plt.grid(True)
plt.show()

# 1. Create the 'models' folder if it doesn't exist
os.makedirs('models', exist_ok=True) 

# 2. Define the path where the model will be saved
model_save_path = 'models/random_forest_model.joblib' 

# 3. Save the trained model object (rf_model)
joblib.dump(rf_model, model_save_path)

print(f"\nSUCCESS: Trained model saved to {model_save_path}")