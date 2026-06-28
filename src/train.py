import os
import warnings
warnings.filterwarnings("ignore")
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models import get_feature_lists, get_preprocessor, get_base_models, get_hyperparameter_grids, build_stacking_ensemble


def train_pipeline(data_path, models_dir):
    """
    Core training pipeline: Loads processed data, splits it, tunes base models,
    trains the stacking ensemble, evaluates and compares all models, and saves the final artifact.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load preprocessed data
    print(f"Loading merged dataset from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Please run preprocess.py first.")
        
    df = pd.read_csv(data_path)
    
    # Get feature columns
    numeric_features, categorical_features = get_feature_lists()
    all_features = numeric_features + categorical_features
    target_col = 'Time_taken(min)'
    
    # Verify columns exist
    missing_cols = [c for c in all_features + [target_col] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in dataset: {missing_cols}")
        
    X = df[all_features]
    y = df[target_col]
    
    # 2. Split dataset into train and test
    print("Splitting dataset into train (80%) and test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # 3. Initialize preprocessor and base models
    preprocessor = get_preprocessor()
    base_models = get_base_models(preprocessor)
    param_grids = get_hyperparameter_grids()
    
    tuned_models = {}
    evaluation_results = []
    
    # 4. Perform Hyperparameter Tuning for each model
    for name, model_pipeline in base_models.items():
        print(f"\n--- Tuning Hyperparameters for {name} ---")
        param_grid = param_grids[name]
        
        # Use RandomizedSearchCV for lightweight and fast optimization
        search = RandomizedSearchCV(
            estimator=model_pipeline,
            param_distributions=param_grid,
            n_iter=5 if name != 'Ridge' else 3, # Keep iterations low for CPU efficiency
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print(f"Running randomized search for {name}...")
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        tuned_models[name] = best_model
        print(f"Best parameters found for {name}: {search.best_params_}")
        
        # Evaluate model on test set
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name} Evaluation: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
        
        evaluation_results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
        
        # Save individual tuned models
        joblib.dump(best_model, os.path.join(models_dir, f"{name}_tuned.joblib"))
        
    # 5. Build and Train the Stacking Ensemble
    print("\n--- Training Stacking Ensemble ---")
    ensemble_pipeline = build_stacking_ensemble(tuned_models)
    
    print("Fitting Stacking Regressor on training set...")
    ensemble_pipeline.fit(X_train, y_train)
    
    # Evaluate ensemble on test set
    y_pred_ensemble = ensemble_pipeline.predict(X_test)
    rmse_ens = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
    mae_ens = mean_absolute_error(y_test, y_pred_ensemble)
    r2_ens = r2_score(y_test, y_pred_ensemble)
    
    print(f"Stacking Ensemble Evaluation: RMSE={rmse_ens:.3f}, MAE={mae_ens:.3f}, R2={r2_ens:.3f}")
    
    evaluation_results.append({
        'Model': 'StackingEnsemble',
        'RMSE': rmse_ens,
        'MAE': mae_ens,
        'R2': r2_ens
    })
    
    # Save the Stacking Ensemble model
    ensemble_path = os.path.join(models_dir, "ensemble_model.joblib")
    joblib.dump(ensemble_pipeline, ensemble_path)
    print(f"Saved final ensemble model to: {ensemble_path}")
    
    # 6. Report and Compare Results
    df_results = pd.DataFrame(evaluation_results)
    print("\n==============================================")
    print("           MODEL COMPARISON SUMMARY           ")
    print("==============================================")
    print(df_results.to_string(index=False))
    print("==============================================")
    
    # Save comparison report to CSV
    df_results.to_csv(os.path.join(models_dir, "model_comparison_metrics.csv"), index=False)
    
    # 7. Extract Feature Importance from LightGBM as Representative Tree-based Model
    try:
        print("\nExtracting and plotting feature importances from LightGBM...")
        lgb_model = tuned_models['LightGBM']
        # Extract preprocessor from pipeline
        fitted_preprocessor = lgb_model.named_steps['preprocessor']
        # Get feature names out of preprocessor
        feature_names = fitted_preprocessor.get_feature_names_out()
        
        # Strip prefixes like 'num__' and 'cat__' from names
        cleaned_feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
        
        importances = lgb_model.named_steps['regressor'].feature_importances_
        
        feat_imp_df = pd.DataFrame({
            'Feature': cleaned_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Plot top 15 features
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', palette='viridis')
        plt.title('Top 15 Most Important Features (LightGBM)')
        plt.xlabel('Importance (Split/Gain)')
        plt.ylabel('Features')
        plt.tight_layout()
        
        feat_imp_path = os.path.join(models_dir, "feature_importance.png")
        plt.savefig(feat_imp_path)
        plt.close()
        print(f"Saved feature importance plot to: {feat_imp_path}")
        
        # 8. Compute and save SHAP plots as images
        try:
            print("\nComputing SHAP values and saving plots...")
            X_test_transformed = fitted_preprocessor.transform(X_test)
            X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=cleaned_feature_names)
            
            explainer = shap.TreeExplainer(lgb_model.named_steps['regressor'])
            shap_values = explainer(X_test_transformed_df)
            
            # Plot 1: SHAP summary graph
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_transformed_df, show=False)
            plt.title("Global Feature Impact (SHAP Summary Plot)", fontsize=14)
            plt.tight_layout()
            shap_summary_path = os.path.join(models_dir, "shap_summary.png")
            plt.savefig(shap_summary_path, bbox_inches='tight')
            plt.close()
            print(f"Saved SHAP summary plot to: {shap_summary_path}")
            
            # Plot 2: SHAP waterfall graph (First sample in test set)
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title("Decision Breakdown for a Single Test Prediction", fontsize=14)
            plt.tight_layout()
            shap_waterfall_path = os.path.join(models_dir, "shap_waterfall.png")
            plt.savefig(shap_waterfall_path, bbox_inches='tight')
            plt.close()
            print(f"Saved SHAP waterfall plot to: {shap_waterfall_path}")
            
        except Exception as shap_err:
            print(f"Could not generate SHAP plots: {shap_err}")
            
    except Exception as e:
        print(f"Could not extract feature importances: {e}")
        
    print("\nTraining run successfully completed!")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'merged_delivery_weather.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    train_pipeline(data_path, models_dir)
