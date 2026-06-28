# %% [markdown]
# # Last-Mile Delivery Delay Prediction: Consolidated Pipeline
# 
# This file combines the entire preprocessing, modeling, training, and prediction pipeline
# into a single script. 
# 
# **How to run cell-by-cell in VS Code:**
# 1. Click "Run Cell" above any cell starting with `# %%`.
# 2. Select the Python environment: `venv` (located in the project folder).

# %% [markdown]
# ## Cell 1: Import Libraries and Define Configurations
# In this cell, we import all required packages and set up path configurations.

# %%
import os
import warnings
warnings.filterwarnings("ignore")
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("Libraries imported successfully!")
print(f"Project directory: {BASE_DIR}")

# %% [markdown]
# ## Cell 2: Verify Datasets
# Check if the raw datasets are placed in `data/raw/`.

# %%
def verify_data():
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not csv_files:
        print("[X] Status: No datasets found in data/raw!")
        print(f"Please place your downloaded Kaggle CSV files inside: {RAW_DIR}")
    else:
        print(f"[OK] Found {len(csv_files)} CSV files in data/raw:")
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")

verify_data()

# %% [markdown]
# ## Cell 3: Data Preprocessing and Weather Integration
# Defines the preprocessing functions, maps cities, merges historical weather, and calculates distance.

# %%
CITY_MAPPING = {
    'BANG': 'bengaluru', 'BEN': 'bengaluru', 'MUM': 'bombay', 'BOM': 'bombay',
    'DEL': 'delhi', 'HYD': 'hyderabad', 'JAIPUR': 'jaipur', 'JAI': 'jaipur',
    'KNP': 'kanpur', 'KAN': 'kanpur', 'NAG': 'nagpur', 'CHEN': 'chennai',
    'CHE': 'chennai', 'KOL': 'kolkata', 'COIMB': 'coimbatore', 'PUNE': 'pune',
    'PUN': 'pune', 'INDO': 'indore', 'IND': 'indore', 'SUR': 'surat',
    'LUDH': 'ludhiana', 'RANCHI': 'ranchi', 'AGRA': 'agra', 'ALH': 'allahabad',
    'AURG': 'aurangabad', 'BHO': 'bhopal', 'BHOP': 'bhopal', 'GOA': 'goa', 'VAD': 'vadodara'
}

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371.0

def process_weather():
    weather_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    weather_files = [f for f in weather_files if 'delivery' not in os.path.basename(f).lower() 
                     and 'train' not in os.path.basename(f).lower() 
                     and 'test' not in os.path.basename(f).lower()]
    if not weather_files:
        return None, None
    all_dfs = []
    for f in weather_files:
        city_name = os.path.splitext(os.path.basename(f))[0].lower()
        df = pd.read_csv(f)
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
            df = df.dropna(subset=['date_time'])
            df['city'] = city_name
            df['month'] = df['date_time'].dt.month
            df['hour'] = df['date_time'].dt.hour
            weather_cols = ['tempC', 'humidity', 'precipMM', 'windspeedKmph', 'pressure', 'cloudcover']
            existing = [c for c in weather_cols if c in df.columns]
            for col in existing:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            all_dfs.append(df.groupby(['city', 'month', 'hour'])[existing].mean().reset_index())
    
    if not all_dfs:
        return None, None
    full_df = pd.concat(all_dfs, ignore_index=True)
    all_features = [c for c in full_df.columns if c not in ['city', 'month', 'hour']]
    national_fallback = full_df.groupby(['month', 'hour'])[all_features].mean().reset_index()
    return full_df, national_fallback

# Preprocess delivery data
print("Running preprocessing...")
delivery_train_paths = glob.glob(os.path.join(RAW_DIR, "*delivery*.csv")) + glob.glob(os.path.join(RAW_DIR, "*train*.csv"))
delivery_train_paths = [p for p in delivery_train_paths if 'weather' not in os.path.basename(p).lower()]

if not delivery_train_paths:
    print("[X] Error: Could not find train.csv in data/raw.")
else:
    df_delivery = pd.read_csv(delivery_train_paths[0])
    for col in df_delivery.columns:
        if df_delivery[col].dtype == 'object':
            df_delivery[col] = df_delivery[col].astype(str).str.strip().replace(['NaN', 'nan'], np.nan)
            
    # Clean target
    target_col = 'Time_taken(min)'
    df_delivery[target_col] = df_delivery[target_col].astype(str).str.replace(r'\(min\)\s*', '', regex=True)
    df_delivery[target_col] = pd.to_numeric(df_delivery[target_col], errors='coerce')
    df_delivery = df_delivery.dropna(subset=[target_col])
    
    # Fill numeric columns
    df_delivery['Delivery_person_Age'] = pd.to_numeric(df_delivery['Delivery_person_Age'], errors='coerce')
    df_delivery['Delivery_person_Age'] = df_delivery['Delivery_person_Age'].fillna(df_delivery['Delivery_person_Age'].median())
    
    df_delivery['Delivery_person_Ratings'] = pd.to_numeric(df_delivery['Delivery_person_Ratings'], errors='coerce')
    df_delivery['Delivery_person_Ratings'] = df_delivery['Delivery_person_Ratings'].fillna(df_delivery['Delivery_person_Ratings'].mean())
    
    df_delivery['multiple_deliveries'] = pd.to_numeric(df_delivery['multiple_deliveries'], errors='coerce').fillna(0.0)
    
    # Clean coordinates
    coord_cols = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']
    for col in coord_cols:
        df_delivery[col] = pd.to_numeric(df_delivery[col], errors='coerce').abs()
    df_delivery = df_delivery.dropna(subset=coord_cols)
    
    # Haversine distance
    df_delivery['distance_km'] = haversine_distance(
        df_delivery['Restaurant_latitude'], df_delivery['Restaurant_longitude'],
        df_delivery['Delivery_location_latitude'], df_delivery['Delivery_location_longitude']
    )
    
    # City extraction
    df_delivery['city_code'] = df_delivery['Delivery_person_ID'].apply(lambda x: x.split('RES')[0].strip() if isinstance(x, str) and 'RES' in x else None)
    df_delivery['mapped_city_name'] = df_delivery['city_code'].map(CITY_MAPPING).fillna('unknown')
    
    # Temporal features
    df_delivery['parsed_date'] = pd.to_datetime(df_delivery['Order_Date'], format='%d-%m-%Y', errors='coerce')
    mask = df_delivery['parsed_date'].isna()
    if mask.any():
        df_delivery.loc[mask, 'parsed_date'] = pd.to_datetime(df_delivery.loc[mask, 'Order_Date'], errors='coerce')
    df_delivery = df_delivery.dropna(subset=['parsed_date'])
    df_delivery['order_month'] = df_delivery['parsed_date'].dt.month
    df_delivery['order_day_of_week'] = df_delivery['parsed_date'].dt.dayofweek
    df_delivery['order_day_of_month'] = df_delivery['parsed_date'].dt.day
    df_delivery['is_weekend'] = df_delivery['order_day_of_week'].isin([5, 6]).astype(int)
    
    def extract_h(x):
        if pd.isna(x) or not isinstance(x, str): return 12.0
        try:
            return float(x.split(':')[0]) if ':' in x else float(x.split('.')[0])
        except:
            return 12.0
    df_delivery['order_hour'] = df_delivery['Time_Orderd'].apply(extract_h)
    
    # Weather merge
    weather_df, national_profile = process_weather()
    if weather_df is not None:
        df_merged = pd.merge(df_delivery, weather_df, left_on=['mapped_city_name', 'order_month', 'order_hour'], right_on=['city', 'month', 'hour'], how='left')
        unmatched = df_merged['tempC'].isna()
        if unmatched.any() and national_profile is not None:
            w_cols = [c for c in weather_df.columns if c not in ['city', 'month', 'hour']]
            unmatched_df = df_merged[unmatched].drop(columns=w_cols + ['city', 'month', 'hour'], errors='ignore')
            matched_fallbacks = pd.merge(unmatched_df, national_profile, left_on=['order_month', 'order_hour'], right_on=['month', 'hour'], how='left')
            matched_cities = df_merged[~unmatched]
            for col in matched_cities.columns:
                if col not in matched_fallbacks.columns:
                    matched_fallbacks[col] = np.nan
            matched_fallbacks = matched_fallbacks[matched_cities.columns]
            df_merged = pd.concat([matched_cities, matched_fallbacks], ignore_index=True)
        df_merged = df_merged.drop(columns=['city', 'month', 'hour'], errors='ignore')
    else:
        df_merged = df_delivery.copy()
        for col in ['tempC', 'humidity', 'precipMM', 'windspeedKmph', 'pressure', 'cloudcover']:
            df_merged[col] = 0.0
            
    weather_features = ['tempC', 'humidity', 'precipMM', 'windspeedKmph', 'pressure', 'cloudcover']
    for col in weather_features:
        df_merged[col] = df_merged[col].fillna(df_merged[col].median() if not pd.isna(df_merged[col].median()) else 0.0)
        
    if 'Weatherconditions' in df_merged.columns:
        df_merged['Weatherconditions'] = df_merged['Weatherconditions'].astype(str).str.replace(r'conditions\s*', '', regex=True).replace('nan', np.nan)
        df_merged['Weatherconditions'] = df_merged['Weatherconditions'].fillna(df_merged['Weatherconditions'].mode()[0])
        
    categorical_cols = ['Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City', 'Weatherconditions']
    for col in categorical_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(df_merged[col].mode()[0] if not df_merged[col].mode().empty else 'unknown').astype(str)
            
    df_merged.to_csv(os.path.join(PROCESSED_DIR, "merged_delivery_weather.csv"), index=False)
    print(f"[OK] Preprocessing finished! Total Rows: {len(df_merged)}")

# %% [markdown]
# ## Cell 4: Train, Tune, and Compare Machine Learning Models
# Set up column preprocessing pipelines, train Random Forest, LightGBM, XGBoost, and Ridge Regression, and stack them.

# %%
df_train = pd.read_csv(os.path.join(PROCESSED_DIR, "merged_delivery_weather.csv"))

numeric_features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries', 'distance_km', 'order_month', 'order_day_of_week', 'order_day_of_month', 'is_weekend', 'order_hour', 'tempC', 'humidity', 'precipMM', 'windspeedKmph', 'pressure', 'cloudcover']
categorical_features = ['Weatherconditions', 'Road_traffic_density', 'Vehicle_condition', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City']
all_features = numeric_features + categorical_features

X = df_train[all_features]
y = df_train['Time_taken(min)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
])

# Base models
base_models = {
    'RandomForest': Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=15))]),
    'LightGBM': Pipeline([('preprocessor', preprocessor), ('regressor', LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1))]),
    'XGBoost': Pipeline([('preprocessor', preprocessor), ('regressor', XGBRegressor(random_state=42, n_jobs=-1))]),
    'CatBoost': Pipeline([('preprocessor', preprocessor), ('regressor', CatBoostRegressor(random_state=42, verbose=0, thread_count=-1, allow_writing_files=False))]),
    'Ridge': Pipeline([('preprocessor', preprocessor), ('regressor', Ridge())])
}

param_grids = {
    'RandomForest': {'regressor__n_estimators': [50, 100], 'regressor__min_samples_split': [2, 10]},
    'LightGBM': {'regressor__n_estimators': [100, 150], 'regressor__learning_rate': [0.05, 0.1]},
    'XGBoost': {'regressor__n_estimators': [100, 150], 'regressor__learning_rate': [0.05, 0.1]},
    'CatBoost': {'regressor__iterations': [100, 150], 'regressor__learning_rate': [0.05, 0.1]},
    'Ridge': {'regressor__alpha': [1.0, 10.0]}
}

tuned_models = {}
results = []

# Tuning
for name, model_pipeline in base_models.items():
    print(f"Tuning {name}...")
    search = RandomizedSearchCV(model_pipeline, param_grids[name], n_iter=3, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    tuned_models[name] = best_model
    
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    print(f"  Best params: {search.best_params_}")
    print(f"  Performance: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

# %% [markdown]
# ## Cell 5: Train and Save Stacking Ensemble
# Combine the tuned base models into a Stacking Regressor and evaluate the final ensemble.

# %%
# Stacking Ensemble
print("Training Stacking Ensemble...")
estimators = [(name, model.named_steps['regressor']) for name, model in tuned_models.items() if name != 'Ridge']
stack_reg = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), n_jobs=-1, cv=3)
ensemble_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', stack_reg)])
ensemble_pipeline.fit(X_train, y_train)

y_pred_ens = ensemble_pipeline.predict(X_test)
rmse_ens = np.sqrt(mean_squared_error(y_test, y_pred_ens))
mae_ens = mean_absolute_error(y_test, y_pred_ens)
r2_ens = r2_score(y_test, y_pred_ens)
results.append({'Model': 'StackingEnsemble', 'RMSE': rmse_ens, 'MAE': mae_ens, 'R2': r2_ens})

print(f"Stacking Ensemble: RMSE={rmse_ens:.3f}, MAE={mae_ens:.3f}, R2={r2_ens:.3f}")

# Print summary comparison table
df_results = pd.DataFrame(results)
print("\n=== MODEL COMPARISON SUMMARY ===")
print(df_results.to_string(index=False))

# Save ensemble
joblib.dump(ensemble_pipeline, os.path.join(MODELS_DIR, "ensemble_model.joblib"))
print("Model saved to models/ensemble_model.joblib")

# %% [markdown]
# ## Cell 6: Visualize Feature Importance
# Plot feature importances from the tuned LightGBM model.

# %%
lgb_model = tuned_models['LightGBM']
feat_names = lgb_model.named_steps['preprocessor'].get_feature_names_out()
cleaned_names = [f.replace('num__', '').replace('cat__', '') for f in feat_names]
importances = lgb_model.named_steps['regressor'].feature_importances_

df_imp = pd.DataFrame({'Feature': cleaned_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_imp.head(15), x='Importance', y='Feature', palette='viridis')
plt.title('Top 15 Feature Importances (LightGBM)')
plt.xlabel('Importance Value')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Cell 7: Explainable AI (XAI) using SHAP
# This cell uses the SHAP library to explain the model's predictions on the test set.
# It outputs:
# 1. A summary plot showing how each feature (including weather parameters) affects the delivery time.
# 2. A waterfall plot explaining a single delivery prediction from the test set.

# %%
import shap
import matplotlib.pyplot as plt
import pandas as pd

# Extract preprocessor and fitted LightGBM from the ensemble pipeline
preprocessor = ensemble_pipeline.named_steps['preprocessor']
lgb_model = ensemble_pipeline.named_steps['regressor'].named_estimators_['LightGBM']

# Transform test data and clean feature names for readability
X_test_transformed = preprocessor.transform(X_test)
feature_names = [f.replace('num__', '').replace('cat__', '') for f in preprocessor.get_feature_names_out()]
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

# Initialize SHAP explainer and compute values
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer(X_test_transformed_df)

# Plot 1: SHAP summary graph (Global Interpretation)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_transformed_df, show=False)
plt.title("Global Feature Impact (SHAP Summary Plot)", fontsize=14)
plt.tight_layout()
plt.show()

# Plot 2: SHAP waterfall graph (Local Interpretation for the first sample in the test set)
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[0], show=False)
plt.title("Decision Breakdown for a Single Test Prediction", fontsize=14)
plt.tight_layout()
plt.show()


