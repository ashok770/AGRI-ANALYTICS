# src/data_processor.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define the relative path to the data
DATA_PATH = os.path.join('archive (5)', 'yield_df.csv')

def load_and_preprocess_data(data_path=DATA_PATH):
    """Loads, cleans, encodes, and splits the data for training."""
    
    # 1. Load Data
    try:
        # Assuming the script is run from the project root or main.py handles the path
        df_yield = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please check your path.")
        return None, None, None, None, None
        
    # 2. Cleanup and Rename
    df = df_yield.drop('Unnamed: 0', axis=1)
    df = df.rename(columns={'hg/ha_yield': 'Yield'})
    
    # 3. One-Hot Encoding
    # Separate features (X) from the target (Y) before encoding
    X_features = df.drop('Yield', axis=1)
    Y_target = df['Yield']
    
    # Encode categorical columns
    df_encoded = pd.get_dummies(X_features, columns=['Area', 'Item'], drop_first=True)
    
    # Handle any remaining NaNs for safety
    df_encoded = df_encoded.dropna()
    Y_target = Y_target.loc[df_encoded.index] # Keep only non-NaN rows in Y

    # Get the final feature list for later use in the web app
    full_feature_list = df_encoded.columns.tolist()

    # 4. Define X and Y
    X = df_encoded
    Y = Y_target

    # 5. Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("\n[Data Processor] Data loaded, cleaned, and split successfully.")
    print(f"   Training set shape: {X_train.shape}")
    
    return X_train, X_test, Y_train, Y_test, full_feature_list