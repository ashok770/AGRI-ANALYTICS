import joblib
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import collections
from sklearn.metrics import mean_absolute_error, r2_score
import os

# --- 1. Define Paths to Saved Assets ---
# Path to the trained model
MODEL_PATH = os.path.join('..', 'models', 'random_forest_model.joblib')
# Path to the saved list of all feature columns
FEATURE_LIST_PATH = os.path.join('..', 'models', 'full_feature_list.joblib') 

try:
    # Load the trained model (The Brain)
    model = joblib.load(MODEL_PATH)
    
    # Load the exact list of 113 features the model was trained on
    FULL_FEATURE_LIST = joblib.load(FEATURE_LIST_PATH) 

    # --- Load the saved X_test and Y_test data for calculating metrics ---
    # NOTE: This assumes you have saved the X_test and Y_test dataframes 
    #       during the main.py execution, which is best practice.
    #       For this solution, we will assume the metrics are available OR
    #       we use the hardcoded values we calculated earlier for the dashboard.
    
    # Placeholder for the actual feature importance data (from your main.py output)
    GLOBAL_TOP_FEATURES = [
        {'feature': 'Item_Potatoes', 'score': 0.369278}, 
        {'feature': 'pesticides_tonnes', 'score': 0.070673},
        {'feature': 'avg_temp', 'score': 0.043567},
        {'feature': 'average_rain_fall_mm_per_year', 'score': 0.042398},
        {'feature': 'Area_India', 'score': 0.056951},
        {'feature': 'Year', 'score': 0.032857}
    ]
    
    # Placeholder for calculated metrics (from your main.py output)
    R2_SCORE = 0.9875
    MAE_SCORE = 3461.19
    
    print(f"Model and Feature List loaded successfully.")
    
except Exception as e:
    print(f"FATAL ERROR: Could not load required model files. Error: {e}")
    model = None
    FULL_FEATURE_LIST = []
    GLOBAL_TOP_FEATURES = []
    R2_SCORE = 0.0
    MAE_SCORE = 0.0


# --- 2. Initialize Flask App ---
app = Flask(__name__)
app.template_folder = os.path.join(os.path.dirname(__file__), 'templates')


# --- 3. Home / Prediction Route ('/') ---
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    explanation_text = None

    if model is None or not FULL_FEATURE_LIST:
        return render_template('base.html', content="<h1>Initialization Error</h1><p>Model files failed to load. Check console for details.</p>")

    if request.method == 'POST':
        try:
            # 3.1. Get and convert data from the form
            data = {
                'Year': int(request.form['Year']),
                'pesticides_tonnes': float(request.form['pesticides_tonnes']),
                'avg_temp': float(request.form['avg_temp']),
                'average_rain_fall_mm_per_year': float(request.form['average_rain_fall_mm_per_year']),
                'Area': request.form['Area'],
                'Item': request.form['Item']
            }

            # 3.2. Prepare the data for the model (One-Hot Encoding)
            input_features_dict = collections.OrderedDict([(feature, 0.0) for feature in FULL_FEATURE_LIST])
            
            # Set continuous numerical features
            input_features_dict['Year'] = data['Year']
            input_features_dict['pesticides_tonnes'] = data['pesticides_tonnes']
            input_features_dict['avg_temp'] = data['avg_temp']
            input_features_dict['average_rain_fall_mm_per_year'] = data['average_rain_fall_mm_per_year']

            # Set ONE-HOT encoded features
            area_col = f"Area_{data['Area']}"
            item_col = f"Item_{data['Item']}"

            if area_col in input_features_dict:
                input_features_dict[area_col] = 1.0
            if item_col in input_features_dict:
                input_features_dict[item_col] = 1.0

            final_input = pd.DataFrame([input_features_dict])
            
            # 3.3. Get Prediction
            predicted_yield = model.predict(final_input)[0]

            # 3.4. Generate Explanation (Based on your Phase 4 results)
            if data['Item'] == 'Potatoes':
                explanation_text = "Analysis: The high yield potential is strongly tied to your **Crop Choice (Potatoes)**, the #1 factor. Evaluate your **Pesticides** use for cost efficiency."
            elif data['Item'] == 'Rice, paddy':
                explanation_text = "Analysis: Your crop choice (**Rice**) is a major factor. The yield will be sensitive to **Rainfall** and **Temperature** in your region this season."
            elif data['Area'] == 'India':
                explanation_text = "Analysis: Your region (**India**) is a primary factor (#5). Focus on optimizing water usage as the yield is highly sensitive to **Rainfall** and **Temperature**."
            else:
                 explanation_text = "Analysis: Yield is heavily influenced by **Temperature and Rainfall**. Check soil health and nutrient application for maximum results."

            prediction_text = f"{predicted_yield:,.2f} hg/ha"

        except ValueError:
            prediction_text = "Error: Please ensure all inputs are valid numbers."
        except Exception as e:
            prediction_text = f"An internal prediction error occurred. (Error: {e})"
            
    # 4. Render the HTML page (using index.html)
    return render_template('index.html', 
                           prediction_text=prediction_text,
                           explanation_text=explanation_text)


# --- 4. Navigation Routes (Aligned Correctly) ---

@app.route('/dashboard')
def dashboard():
    # Pass the hardcoded metrics and features (from main.py output) to the dashboard template
    return render_template('dashboard.html', 
                           r2_score=R2_SCORE, 
                           mae_score=MAE_SCORE, 
                           features=GLOBAL_TOP_FEATURES)

# web_app/app.py (Updated /what-if route)

# NOTE: This uses the existing model, FULL_FEATURE_LIST, and collections import

@app.route('/what-if', methods=['GET', 'POST'])
def whatif():
    scenario_results = None
    variable_factor = None

    if model is None or not FULL_FEATURE_LIST:
        return render_template('base.html', content="<h1>Initialization Error</h1><p>Model files failed to load. Check console for details.</p>")

    if request.method == 'POST':
        try:
            # 1. Get constant inputs
            const_data = {
                'Year': int(request.form['Year']),
                'Area': request.form['Area'],
                'Item': request.form['Item'],
            }

            # 2. Get scenario inputs
            variable_factor = request.form['variable_factor']
            start_val = float(request.form['start_value'])
            end_val = float(request.form['end_value'])
            steps = int(request.form['steps'])
            
            # Create the range of values to test
            test_values = np.linspace(start_val, end_val, steps)

            results = []
            base_yield = None

            for val in test_values:
                # 3. Build the input dictionary for THIS specific test value
                input_features_dict = collections.OrderedDict([(feature, 0.0) for feature in FULL_FEATURE_LIST])
                
                # Set constants
                input_features_dict['Year'] = const_data['Year']
                # Set ONE-HOT constants
                if f"Area_{const_data['Area']}" in input_features_dict: input_features_dict[f"Area_{const_data['Area']}"] = 1.0
                if f"Item_{const_data['Item']}" in input_features_dict: input_features_dict[f"Item_{const_data['Item']}"] = 1.0
                
                # Set the VARIABLE factor and give dummy constant values to non-varied inputs
                if variable_factor == 'pesticides_tonnes':
                    input_features_dict[variable_factor] = val
                    input_features_dict['avg_temp'] = 20.0 # Dummy Avg Temp
                    input_features_dict['average_rain_fall_mm_per_year'] = 1500.0 # Dummy Rainfall
                elif variable_factor == 'avg_temp':
                    input_features_dict[variable_factor] = val
                    input_features_dict['pesticides_tonnes'] = 1000.0
                    input_features_dict['average_rain_fall_mm_per_year'] = 1500.0
                elif variable_factor == 'average_rain_fall_mm_per_year':
                    input_features_dict[variable_factor] = val
                    input_features_dict['pesticides_tonnes'] = 1000.0
                    input_features_dict['avg_temp'] = 20.0
                
                # 4. Predict and store results
                final_input = pd.DataFrame([input_features_dict])
                predicted_yield = model.predict(final_input)[0]

                if base_yield is None:
                    base_yield = predicted_yield # Set the yield for the starting value

                change_percent = ((predicted_yield - base_yield) / base_yield) * 100

                results.append({
                    'factor_value': f"{val:.2f}",
                    'predicted_yield': f"{predicted_yield:,.2f}",
                    'change_percent': change_percent
                })
            
            scenario_results = results

        except ValueError:
            return render_template('whatif.html', error_message="Error: Please ensure all inputs are valid numbers.")
        except Exception as e:
            return render_template('whatif.html', error_message=f"An error occurred: {e}")
            
    return render_template('whatif.html', scenario_results=scenario_results, variable_factor=variable_factor)

@app.route('/about')
def about():
    return render_template('about.html')


# --- 5. Run the App ---
if __name__ == '__main__':
    app.run()