"""
model_comparison.py

Usage:
    python model_comparison.py /path/to/liquid_class_parameters_filtered.csv

Outputs:
 - Prints the optimized XGBoost model's RMSE.
 - Prints a detailed comparison for each test data point.
 - Exports the trained XGBoost models to a file named 'xgb_models.joblib'.
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import joblib

# ---------- Configuration ----------
RANDOM_SEED = 42
TEST_SIZE = 40  # Number of data points to hold out for testing
HIDDEN_PROPORTION = 0.5  # Proportion of parameters to "hide" for the autocomplete test
MODEL_EXPORT_PATH = 'xgb_models.joblib'
# -----------------------------------

np.random.seed(RANDOM_SEED)

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Loads CSV, drops non-numeric and unique columns, and handles NaNs."""
    df = pd.read_csv(path)
    if 'LiquidClassName' in df.columns:
        df = df.drop(columns=['LiquidClassName'])
    
    for col in df.columns:
        if df[col].nunique() == len(df[col]):
            print(f"Dropping unique identifier column: {col}")
            df = df.drop(columns=[col])

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Converting categorical column {col} to numeric codes.")
            df[col], _ = pd.factorize(df[col])

    df = df.fillna(df.median())
    return df

def fit_xgboost_models(df_train: pd.DataFrame, cols: list):
    """
    Fits an XGBoost model for each column, treating it as a regression problem
    with all other columns as features. This version includes Bayesian Optimization.
    """
    xgb_models = {}
    
    # Define the search space for Bayesian Optimization
    search_space = {
        'n_estimators': Integer(50, 500),
        'learning_rate': Real(0.01, 0.2, 'log-uniform'),
        'max_depth': Integer(3, 10),
        'subsample': Real(0.5, 1.0, 'uniform'),
        'colsample_bytree': Real(0.5, 1.0, 'uniform'),
        'gamma': Real(0, 10, 'uniform'),
        'min_child_weight': Integer(1, 10)
    }

    for target_col in cols:
        print(f"Training and optimizing XGBoost model for target: {target_col}")
        features = [col for col in cols if col != target_col]
        X_train = df_train[features]
        y_train = df_train[target_col]
        
        # Initialize the base model
        model = XGBRegressor(objective='reg:squarederror', random_state=RANDOM_SEED)

        # Initialize BayesSearchCV for hyperparameter optimization
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=10,  # Number of optimization iterations
            cv=5,       # 5-fold cross-validation
            n_jobs=-1,  # Use all available cores
            scoring='neg_mean_squared_error',
            random_state=RANDOM_SEED
        )
        
        # Perform the search
        bayes_search.fit(X_train, y_train)
        
        # Store the best model found
        xgb_models[target_col] = bayes_search.best_estimator_
        print(f"Best hyperparameters for {target_col}: {bayes_search.best_params_}")
        
    return xgb_models

def get_xgboost_recommendation(xgb_models, test_point_original, cols_hidden, cols_observed):
    """
    Uses the trained XGBoost models to predict the hidden values.
    """
    xgb_recommendation = pd.Series(index=cols_hidden, dtype=float)
    
    for target_col in cols_hidden:
        model = xgb_models[target_col]
        
        # Create a feature set from the observed values of the test point
        features_for_this_model = [col for col in test_point_original.index if col != target_col]
        X_test = test_point_original[features_for_this_model].values.reshape(1, -1)
        
        # Predict the hidden value
        prediction = model.predict(X_test)[0]
        xgb_recommendation[target_col] = prediction
        
    return xgb_recommendation

def main(csv_path: str):
    print("Starting XGBoost Autorecommendation with Bayesian Optimization.")
    
    # Ensure scikit-optimize is installed
    try:
        from skopt import BayesSearchCV
    except ImportError:
        print("Scikit-optimize (skopt) is not installed. Please install it using: pip install scikit-optimize")
        sys.exit(1)

    df = load_and_clean_data(csv_path)
    print("Cleaned data shape:", df.shape)

    if df.shape[0] < TEST_SIZE + 10:
        print("Dataset is too small for a meaningful comparison.")
        return

    # Split data into training and test sets
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    # Get column names
    cols = df.columns.tolist()
    
    print("\n--- Training and Optimizing XGBoost Models ---")
    xgb_models = fit_xgboost_models(df_train, cols)
    
    # Export the trained models
    print(f"\n--- Exporting Models to {MODEL_EXPORT_PATH} ---")
    joblib.dump(xgb_models, MODEL_EXPORT_PATH)
    print("Models exported successfully.")

    print("\n--- Running Autorecommendation Test ---")
    xgb_errors = []
    
    for i in range(TEST_SIZE):
        test_point_original = df_test.iloc[i].copy()
        
        # Randomly hide a proportion of the parameters
        cols_hidden = np.random.choice(cols, int(len(cols) * HIDDEN_PROPORTION), replace=False)
        
        # Get recommendations from the optimized XGBoost models
        xgb_rec = get_xgboost_recommendation(xgb_models, test_point_original, cols_hidden, None) # cols_observed not used here
        
        # Calculate overall RMSE for this test point
        xgb_rmse = np.sqrt(mean_squared_error(test_point_original[cols_hidden], xgb_rec))
        xgb_errors.append(xgb_rmse)
        
        print(f"\n--- Test Point {i+1} ---")
        print(f"Overall XGBoost RMSE: {xgb_rmse:.2f}")
        print("\nIndividual Parameter RMSE:")
        
        for param in cols_hidden:
            xgb_param_rmse = np.sqrt(mean_squared_error([test_point_original[param]], [xgb_rec[param]]))
            print(f"  - {param}: XGBoost RMSE = {xgb_param_rmse:.2f}")

    # Final summary
    print("\n--- Final Results ---")
    print(f"Average XGBoost RMSE: {np.mean(xgb_errors):.4f}")
    
    print("\nOptimization and analysis complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_comparison.py /path/to/liquid_class_parameters_filtered.csv")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)