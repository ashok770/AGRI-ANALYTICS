# main.py

from src.data_processor import load_and_preprocess_data
from src.model_trainer import train_and_save_model
from src.evaluator import generate_feature_importance, print_performance_metrics
import joblib
import os
if __name__ == "__main__":
    print("--- Starting Crop Yield Prediction Pipeline ---")
    
    # 1. DATA PROCESSING (Calls data_processor.py)
    X_train, X_test, Y_train, Y_test, FULL_FEATURE_LIST = load_and_preprocess_data()
    
    if X_train is None:
        print("Pipeline aborted due to data loading error.")
    else:
        FULL_FEATURE_LIST_PATH = 'models/full_feature_list.joblib' 
        joblib.dump(FULL_FEATURE_LIST, FULL_FEATURE_LIST_PATH)
        print(f"Feature list saved to {FULL_FEATURE_LIST_PATH}")
        

        # 2. MODEL TRAINING (Calls model_trainer.py)
        rf_model, Y_pred = train_and_save_model(X_train, X_test, Y_train, Y_test)
        
        # 3. EVALUATION AND INSIGHTS (Calls evaluator.py)
        print_performance_metrics(Y_test, Y_pred)
        top_features = generate_feature_importance(rf_model, X_train)
        
        # NOTE: You would typically save the FULL_FEATURE_LIST here for the web_app
        # For instance: joblib.dump(FULL_FEATURE_LIST, 'models/feature_list.joblib')
        
        print("\n--- Pipeline Complete. Model is ready to deploy! ---")


        