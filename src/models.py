from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

def get_feature_lists():
    """Returns the names of numerical and categorical features."""
    numeric_features = [
        'Delivery_person_Age',
        'Delivery_person_Ratings',
        'multiple_deliveries',
        'distance_km',
        'order_month',
        'order_day_of_week',
        'order_day_of_month',
        'is_weekend',
        'order_hour',
        # Weather-Integrated Climatological Features
        'tempC',
        'humidity',
        'precipMM',
        'windspeedKmph',
        'pressure',
        'cloudcover'
    ]
    
    categorical_features = [
        'Weatherconditions',
        'Road_traffic_density',
        'Vehicle_condition',  # Can be categorical/ordinal
        'Type_of_order',
        'Type_of_vehicle',
        'Festival',
        'City'
    ]
    
    return numeric_features, categorical_features

def get_preprocessor():
    """Builds a scikit-learn ColumnTransformer for preprocessing numeric and categorical data."""
    numeric_features, categorical_features = get_feature_lists()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return preprocessor

def get_base_models(preprocessor):
    """
    Initializes base machine learning pipelines.
    Each model includes the preprocessor inside its pipeline to prevent data leakage.
    """
    models = {
        'RandomForest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=15, n_estimators=100))
        ]),
        'LightGBM': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, n_estimators=150, learning_rate=0.05))
        ]),
        'XGBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(random_state=42, n_jobs=-1, n_estimators=150, learning_rate=0.05, max_depth=6))
        ]),
        'CatBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', CatBoostRegressor(random_state=42, verbose=0, thread_count=-1, iterations=150, learning_rate=0.05, allow_writing_files=False))
        ]),
        'Ridge': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', Ridge(alpha=1.0))
        ])
    }
    return models

def get_hyperparameter_grids():
    """Returns parameter distributions for randomized hyperparameter tuning."""
    param_grids = {
        'RandomForest': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [10, 15, 20, None],
            'regressor__min_samples_split': [2, 5, 10]
        },
        'LightGBM': {
            'regressor__n_estimators': [100, 150, 200],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__num_leaves': [15, 31, 63],
            'regressor__max_depth': [-1, 5, 10]
        },
        'XGBoost': {
            'regressor__n_estimators': [100, 150, 200],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 6, 9],
            'regressor__subsample': [0.8, 1.0]
        },
        'CatBoost': {
            'regressor__iterations': [100, 150, 200],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__depth': [4, 6, 8]
        },
        'Ridge': {
            'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
        }
    }
    return param_grids

def build_stacking_ensemble(tuned_models):
    """
    Builds a Stacking Regressor ensemble combining the tuned base models.
    Uses Ridge Regression as the final meta-regressor to blend their outputs.
    """
    # Exclude Ridge from stacking base models for better stacking diversity
    estimators = []
    for name, model in tuned_models.items():
        if name != 'Ridge':
            # Extract the regressor steps directly to pass to StackingRegressor
            # StackingRegressor will receive the raw inputs and we pass the ColumnTransformer preprocessor
            # through the stacking regressor pipeline.
            # StackingRegressor expects list of tuples: (name, estimator)
            estimators.append((name, model.named_steps['regressor']))
            
    preprocessor = tuned_models['LightGBM'].named_steps['preprocessor']
    
    # Define stacking regressor
    stack_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1,
        cv=5
    )
    
    # Wrap inside pipeline with preprocessor
    ensemble_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', stack_reg)
    ])
    
    return ensemble_pipeline
