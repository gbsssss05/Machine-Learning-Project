import os
import glob
import pandas as pd
import numpy as np
from geopy.distance import geodesic

# Dictionary to map city codes from Delivery_person_ID to weather dataset filenames
CITY_MAPPING = {
    'BANG': 'bengaluru',
    'BEN': 'bengaluru',
    'MUM': 'bombay',
    'BOM': 'bombay',
    'DEL': 'delhi',
    'HYD': 'hyderabad',
    'JAIPUR': 'jaipur',
    'JAI': 'jaipur',
    'KNP': 'kanpur',
    'KAN': 'kanpur',
    'NAG': 'nagpur',
    'CHEN': 'chennai',
    'CHE': 'chennai',
    'KOL': 'kolkata',
    'COIMB': 'coimbatore',
    'PUNE': 'pune',
    'PUN': 'pune',
    'INDO': 'indore',
    'IND': 'indore',
    'SUR': 'surat',
    'LUDH': 'ludhiana',
    'RANCHI': 'ranchi',
    'AGRA': 'agra',
    'ALH': 'allahabad',
    'AURG': 'aurangabad',
    'BHO': 'bhopal',
    'BHOP': 'bhopal',
    'GOA': 'goa',
    'VAD': 'vadodara'
}

def extract_city_code(dp_id):
    """Extracts the city abbreviation prefix from Delivery_person_ID."""
    if not isinstance(dp_id, str):
        return None
    # IDs look like 'BANGRES01DEL01' or 'INDORES13DEL02'
    parts = dp_id.split('RES')
    if len(parts) > 1:
        return parts[0].strip()
    return None

def extract_hour(time_str):
    """Robustly extracts the hour integer from time strings like HH:MM or HH:MM:SS."""
    if pd.isna(time_str) or not isinstance(time_str, str):
        return np.nan
    time_str = time_str.strip()
    try:
        if ':' in time_str:
            return int(time_str.split(':')[0])
        elif '.' in time_str:
            return int(time_str.split('.')[0])
        else:
            return int(time_str)
    except:
        return np.nan

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates geodistance in kilometers using Haversine formula."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0  # Radius of earth in kilometers
    return c * r

def process_historical_weather(raw_dir):
    """
    Loads all weather CSV files, aggregates them by city, month, and hour,
    and returns a weather profile lookup table along with a national fallback profile.
    """
    weather_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    
    # Filter files that are not the main delivery dataset files
    weather_files = [f for f in weather_files if 'delivery' not in os.path.basename(f).lower() 
                     and 'train' not in os.path.basename(f).lower() 
                     and 'test' not in os.path.basename(f).lower()]
    
    if not weather_files:
        print("Warning: No city weather files found. Weather integration will rely on fallback data.")
        return None, None
        
    all_weather_dfs = []
    
    print("Loading and aggregating historical weather files...")
    for fpath in weather_files:
        filename = os.path.basename(fpath)
        city_name = os.path.splitext(filename)[0].lower() # e.g. bengaluru
        
        try:
            # Read weather file (hourly data)
            df = pd.read_csv(fpath)
            # Ensure required columns exist
            if 'date_time' not in df.columns:
                print(f"Skipping {filename}: 'date_time' column missing.")
                continue
                
            df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
            df = df.dropna(subset=['date_time'])
            
            df['city'] = city_name
            df['month'] = df['date_time'].dt.month
            df['hour'] = df['date_time'].dt.hour
            
            # Select relevant weather features to aggregate
            weather_cols = ['tempC', 'humidity', 'precipMM', 'windspeedKmph', 'pressure', 'cloudcover']
            existing_cols = [col for col in weather_cols if col in df.columns]
            
            # Convert columns to numeric
            for col in existing_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Aggregate hourly weather features by city, month, and hour
            grouped = df.groupby(['city', 'month', 'hour'])[existing_cols].mean().reset_index()
            all_weather_dfs.append(grouped)
            print(f"Loaded and aggregated {len(grouped)} weather profiles for city: '{city_name}'")
            
        except Exception as e:
            print(f"Error processing weather file {filename}: {e}")
            
    if not all_weather_dfs:
        return None, None
        
    full_weather_df = pd.concat(all_weather_dfs, ignore_index=True)
    
    # Calculate national monthly-hourly average profile across all available cities as a fallback
    all_cols = [c for c in full_weather_df.columns if c not in ['city', 'month', 'hour']]
    national_profile = full_weather_df.groupby(['month', 'hour'])[all_cols].mean().reset_index()
    
    return full_weather_df, national_profile

def preprocess_pipeline(raw_dir, processed_dir):
    """
    Core data preprocessing pipeline. Loads, cleans, feature engineers, and merges
    the delivery dataset and weather dataset.
    """
    # 1. Look for food delivery dataset (usually train.csv)
    delivery_train_paths = glob.glob(os.path.join(raw_dir, "*delivery*.csv")) + \
                           glob.glob(os.path.join(raw_dir, "*train*.csv"))
    
    # Filter out anything that contains weather or other labels
    delivery_train_paths = [p for p in delivery_train_paths if 'weather' not in os.path.basename(p).lower()]
    
    if not delivery_train_paths:
        raise FileNotFoundError(f"Could not find the food delivery dataset CSV (e.g. train.csv) in {raw_dir}")
        
    delivery_path = delivery_train_paths[0]
    print(f"Loading food delivery dataset from: {delivery_path}")
    df_delivery = pd.read_csv(delivery_path)
    
    # Clean string NaN values across all object columns
    for col in df_delivery.columns:
        if df_delivery[col].dtype == 'object':
            df_delivery[col] = df_delivery[col].astype(str).str.strip()
            df_delivery[col] = df_delivery[col].replace('NaN', np.nan)
            df_delivery[col] = df_delivery[col].replace('nan', np.nan)
            
    # Clean target variable: Time_taken(min)
    target_col = 'Time_taken(min)'
    if target_col not in df_delivery.columns:
        # Check if there is a similar target col
        cols = [c for c in df_delivery.columns if 'time' in c.lower()]
        if cols:
            target_col = cols[0]
        else:
            raise KeyError("Target column (e.g. Time_taken(min)) not found in delivery dataset.")
            
    df_delivery[target_col] = df_delivery[target_col].astype(str).str.replace(r'\(min\)\s*', '', regex=True)
    df_delivery[target_col] = pd.to_numeric(df_delivery[target_col], errors='coerce')
    df_delivery = df_delivery.dropna(subset=[target_col])
    
    # Clean other numeric columns
    df_delivery['Delivery_person_Age'] = pd.to_numeric(df_delivery['Delivery_person_Age'], errors='coerce')
    df_delivery['Delivery_person_Ratings'] = pd.to_numeric(df_delivery['Delivery_person_Ratings'], errors='coerce')
    df_delivery['multiple_deliveries'] = pd.to_numeric(df_delivery['multiple_deliveries'], errors='coerce')
    
    # Fill numeric NaNs with median/mean
    df_delivery['Delivery_person_Age'] = df_delivery['Delivery_person_Age'].fillna(df_delivery['Delivery_person_Age'].median())
    df_delivery['Delivery_person_Ratings'] = df_delivery['Delivery_person_Ratings'].fillna(df_delivery['Delivery_person_Ratings'].mean())
    df_delivery['multiple_deliveries'] = df_delivery['multiple_deliveries'].fillna(0.0)
    
    # Clean coordinates
    coord_cols = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']
    for col in coord_cols:
        df_delivery[col] = pd.to_numeric(df_delivery[col], errors='coerce').abs()
        
    # Drop rows with invalid coordinates
    df_delivery = df_delivery.dropna(subset=coord_cols)
    df_delivery = df_delivery[(df_delivery['Restaurant_latitude'] > 5) & (df_delivery['Restaurant_latitude'] < 40)]
    df_delivery = df_delivery[(df_delivery['Restaurant_longitude'] > 60) & (df_delivery['Restaurant_longitude'] < 100)]
    
    # Feature Engineering: Calculate Geodistance
    print("Calculating geodesic distances between restaurants and delivery points...")
    # Using vectorized Haversine formula for efficiency
    df_delivery['distance_km'] = haversine_distance(
        df_delivery['Restaurant_latitude'], df_delivery['Restaurant_longitude'],
        df_delivery['Delivery_location_latitude'], df_delivery['Delivery_location_longitude']
    )
    
    # Extract City from Delivery_person_ID
    df_delivery['city_code'] = df_delivery['Delivery_person_ID'].apply(extract_city_code)
    df_delivery['mapped_city_name'] = df_delivery['city_code'].map(CITY_MAPPING).fillna('unknown')
    
    # Extract Temporal Features
    print("Extracting temporal features...")
    # Dates are usually formatted as DD-MM-YYYY
    df_delivery['parsed_date'] = pd.to_datetime(df_delivery['Order_Date'], format='%d-%m-%Y', errors='coerce')
    # Fallback parsing in case of varying formats
    mask = df_delivery['parsed_date'].isna()
    if mask.any():
        df_delivery.loc[mask, 'parsed_date'] = pd.to_datetime(df_delivery.loc[mask, 'Order_Date'], errors='coerce')
        
    df_delivery = df_delivery.dropna(subset=['parsed_date'])
    df_delivery['order_month'] = df_delivery['parsed_date'].dt.month
    df_delivery['order_day_of_week'] = df_delivery['parsed_date'].dt.dayofweek
    df_delivery['order_day_of_month'] = df_delivery['parsed_date'].dt.day
    df_delivery['is_weekend'] = df_delivery['order_day_of_week'].isin([5, 6]).astype(int)
    
    # Extract Hour
    df_delivery['order_hour'] = df_delivery['Time_Orderd'].apply(extract_hour)
    df_delivery['order_hour'] = df_delivery['order_hour'].fillna(12.0) # Median fallback
    
    # Load and merge weather data
    weather_df, national_profile = process_historical_weather(raw_dir)
    
    if weather_df is not None:
        print("Integrating weather dataset features...")
        # 1. Merge by City-Month-Hour
        df_merged = pd.merge(
            df_delivery,
            weather_df,
            left_on=['mapped_city_name', 'order_month', 'order_hour'],
            right_on=['city', 'month', 'hour'],
            how='left'
        )
        
        # 2. For unmatched rows (e.g. unknown cities), merge with national average profile
        unmatched_mask = df_merged['tempC'].isna()
        if unmatched_mask.any():
            print(f"Applying national weather fallbacks for {unmatched_mask.sum()} entries...")
            weather_feature_cols = [c for c in weather_df.columns if c not in ['city', 'month', 'hour']]
            
            # Drop the NaN columns for the unmatched rows before merging
            unmatched_df = df_merged[unmatched_mask].drop(columns=weather_feature_cols + ['city', 'month', 'hour'], errors='ignore')
            
            # Merge unmatched rows with national profile
            matched_fallbacks = pd.merge(
                unmatched_df,
                national_profile,
                left_on=['order_month', 'order_hour'],
                right_on=['month', 'hour'],
                how='left'
            )
            
            # Combine the successfully city-matched rows and fallback-matched rows
            matched_cities_df = df_merged[~unmatched_mask]
            
            # Ensure both dataframes have exactly the same columns
            for col in matched_cities_df.columns:
                if col not in matched_fallbacks.columns:
                    matched_fallbacks[col] = np.nan
            matched_fallbacks = matched_fallbacks[matched_cities_df.columns]
            
            df_merged = pd.concat([matched_cities_df, matched_fallbacks], ignore_index=True)
            
        # Clean merge side-effect columns
        df_merged = df_merged.drop(columns=['city', 'month', 'hour'], errors='ignore')
    else:
        print("No weather data integrated. Creating dummy columns for pipeline stability.")
        df_merged = df_delivery.copy()
        for col in ['tempC', 'humidity', 'precipMM', 'windspeedKmph', 'pressure', 'cloudcover']:
            df_merged[col] = 0.0
            
    # Impute remaining missing weather features (if any) with dataset medians
    weather_features = ['tempC', 'humidity', 'precipMM', 'windspeedKmph', 'pressure', 'cloudcover']
    for col in weather_features:
        df_merged[col] = df_merged[col].fillna(df_merged[col].median() if not pd.isna(df_merged[col].median()) else 0.0)
        
    # Clean Categorical Weather Conditions from food delivery dataset
    if 'Weatherconditions' in df_merged.columns:
        df_merged['Weatherconditions'] = df_merged['Weatherconditions'].astype(str).str.replace(r'conditions\s*', '', regex=True)
        df_merged['Weatherconditions'] = df_merged['Weatherconditions'].replace('nan', np.nan)
        df_merged['Weatherconditions'] = df_merged['Weatherconditions'].fillna(df_merged['Weatherconditions'].mode()[0])
        
    # Standardize other categorical fields
    categorical_cols = ['Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City', 'Weatherconditions']
    for col in categorical_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(df_merged[col].mode()[0] if not df_merged[col].mode().empty else 'unknown')
            df_merged[col] = df_merged[col].astype(str)
            
    # Save the processed data
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "merged_delivery_weather.csv")
    df_merged.to_csv(out_path, index=False)
    print(f"\nPreprocessing finished! Merged dataset saved to: {out_path}")
    print(f"Total Rows: {len(df_merged)}")
    print(f"Weather-Integrated Columns: {weather_features}")
    
    return df_merged

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data', 'raw')
    processed_path = os.path.join(base_dir, 'data', 'processed')
    try:
        preprocess_pipeline(raw_path, processed_path)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        print("Note: The pipeline requires raw CSV files to be present in data/raw.")
