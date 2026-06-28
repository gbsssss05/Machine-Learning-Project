import os
import joblib
import pandas as pd
import numpy as np

def predict_delay(input_data, model_path):
    """
    Loads the trained ensemble model and makes a prediction on a dictionary of input parameters.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")
        
    # Load model pipeline
    model = joblib.load(model_path)
    
    # Convert input dict to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Fill in any missing columns with sensible defaults or nan
    # Check what features the pipeline expects
    preprocessor = model.named_steps['preprocessor']
    numeric_features = preprocessor.transformers_[0][2]
    categorical_features = preprocessor.transformers_[1][2]
    all_expected_features = list(numeric_features) + list(categorical_features)
    
    for col in all_expected_features:
        if col not in df_input.columns:
            if col in numeric_features:
                df_input[col] = np.nan
            else:
                df_input[col] = 'unknown'
                
    # Select expected columns in correct order
    df_input = df_input[all_expected_features]
    
    # Predict
    predicted_time = model.predict(df_input)[0]
    return predicted_time

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'ensemble_model.joblib')
    
    # Sample Query
    sample_delivery_query = {
        # Agent profiles
        'Delivery_person_Age': 25.0,
        'Delivery_person_Ratings': 4.7,
        'multiple_deliveries': 1.0,
        'Vehicle_condition': 1,
        'Type_of_vehicle': 'motorcycle',
        'Type_of_order': 'Meal',
        'Festival': 'No',
        'City': 'Urban',
        
        # Geospatial
        'distance_km': 4.5,
        
        # Temporal
        'order_month': 3,
        'order_day_of_week': 2, # Wednesday
        'order_day_of_month': 15,
        'is_weekend': 0,
        'order_hour': 19.0, # 7:00 PM
        
        # Delivery weather environment variables (from historical mapping)
        'Weatherconditions': 'Sunny',
        'tempC': 28.0,
        'humidity': 62.0,
        'precipMM': 0.0,
        'windspeedKmph': 12.0,
        'pressure': 1012.0,
        'cloudcover': 20.0
    }
    
    print("--- Running Single Delivery Prediction Demonstration ---")
    print(f"Sample Input Query Details:")
    for k, v in sample_delivery_query.items():
        print(f"  {k}: {v}")
        
    try:
        predicted_minutes = predict_delay(sample_delivery_query, model_path)
        print("\n==============================================")
        print(f"Predicted Food Delivery Time: {predicted_minutes:.1f} minutes")
        print("==============================================")
    except FileNotFoundError:
        print("\nError: The model files have not been trained yet.")
        print(f"Please place your Kaggle datasets in data/raw and run: 'python src/train.py'")
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
