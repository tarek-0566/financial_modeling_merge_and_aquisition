# ----------------------------------------------------
# Metin Vural 06.01.2025
# ----------------------------------------------------

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, median_absolute_error  #, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# from statsmodels.tsa.seasonal import STL
# import warnings

# # Suppress warnings for cleaner output
# warnings.filterwarnings("ignore")

# ----------------------------------------------------
# 1) Configuration Options
# ----------------------------------------------------

# Choose one model type to train. Options:
# 'knn', 'linear_regression', 'random_forest', 'xgboost',
# 'lasso', 'ridge', 'svr', 'gpr'
model_type = 'knn'  # Change this to your desired model

# Toggle to include or exclude extra (engineered) features
INCLUDE_EXTRA_FEATURES = True  # Set to False to exclude extra features

# ----------------------------------------------------
# 2) Adjust Paths/Folders Dynamically
# ----------------------------------------------------

CURRENT_DIR = os.getcwd()
TARGET_CURRENCY = "EUR"  # Change to "USD" as needed

# Single flag to control both input and output folders
USE_DEFAULT_FOLDERS = False  # True = data & output, False = data_eur & output_eur

# Determine input folder
if USE_DEFAULT_FOLDERS:
    INPUT_FOLDER = os.path.join(CURRENT_DIR, "data")
    OUTPUT_BASE_FOLDER = os.path.join(CURRENT_DIR, "output")
else:
    INPUT_FOLDER = os.path.join(CURRENT_DIR, f'data_{TARGET_CURRENCY.lower()}')
    OUTPUT_BASE_FOLDER = os.path.join(CURRENT_DIR, f'output_{TARGET_CURRENCY.lower()}')

print("Using INPUT_FOLDER:", INPUT_FOLDER)
print("Using OUTPUT_FOLDER:", OUTPUT_BASE_FOLDER)

# Model-specific folders
def get_model_folders(model_type):
    """
    Returns three paths:
     - model_folder: where .pkl files (trained models, scalers) go
     - predictions_folder: where training-time plots/predictions go
     - summary_results_path: CSV of training metrics
    """
    base_folder = os.path.join(OUTPUT_BASE_FOLDER, model_type)
    model_folder = os.path.join(base_folder, 'models')
    predictions_folder = os.path.join(base_folder, 'predictions')
    summary_results_path = os.path.join(base_folder, f'summary_results_{model_type}.csv')
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(predictions_folder, exist_ok=True)
    return model_folder, predictions_folder, summary_results_path

# ----------------------------------------------------
# 3) Choose Which Files to Train On
# ----------------------------------------------------
FILE_LIST = [
    "02_capital_expenditures",
    "03_cash_short_term_inv",
    "04_EBIT",
    "05_EBITDA",
    "06_Free_cash_flow",
    "07_Gross_income",
    "08_Net Debt",
    "09_Net_Fin_Cash_Flow",
    "10_Net_income",
    "11_Net_Inv_cash_flow",
    "12_Net_Op_Cash_Flow",
    "13_Sales",
    "14_Total_assets",
    "15_Total_debt",
    "16_Total_Liabilities",
    "17_Total_sharehold_eq"
]

# ----------------------------------------------------
# 4) Define Base Features and Target
# ----------------------------------------------------
BASE_FEATURES = ['DEC 2019', 'DEC 2020', 'DEC 2021', 'DEC 2022']
TARGET_COLUMN = 'DEC 2023'

# ----------------------------------------------------
# 5) Feature Engineering Functions
# ----------------------------------------------------
from sklearn.impute import SimpleImputer

def handle_missing_values(df):
    """
    Handle missing values by imputing them with the mean of each numeric column.
    Non-numeric columns are left unchanged.
    The original column order is preserved.
    """
    # Identify numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Initialize the imputer with 'mean' strategy
    imputer = SimpleImputer(strategy='mean')
    
    # Impute only numeric columns
    df_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
    
    # Reconstruct the DataFrame with columns in the original order
    df_imputed = pd.concat([df_numeric, df[non_numeric_cols]], axis=1)
    df_imputed = df[df.columns]  # Reorder columns to match original DataFrame
    
    return df_imputed



def create_statistical_features(df, base_features):
    """
    Create statistical summary features across base features.
    """
    df['feat_mean_base'] = df[base_features].mean(axis=1)
    df['feat_std_base'] = df[base_features].std(axis=1)
    df['feat_max_base'] = df[base_features].max(axis=1)
    df['feat_min_base'] = df[base_features].min(axis=1)
    df['feat_range_base'] = df['feat_max_base'] - df['feat_min_base']
    df['feat_sum_base'] = df[base_features].sum(axis=1)
    return df

def create_year_over_year_changes(df, base_features):
    """
    Create features representing year-over-year changes.
    """
    for i in range(1, len(base_features)):
        current_year = base_features[i]
        previous_year = base_features[i-1]
        df[f'feat_change_{current_year}_vs_{previous_year}'] = df[current_year] - df[previous_year]
    return df

def create_growth_rates(df, base_features):
    """
    Create growth rate features between consecutive years.
    """
    for i in range(1, len(base_features)):
        current_year = base_features[i]
        previous_year = base_features[i-1]
        # Avoid division by zero
        df[f'feat_growth_rate_{current_year}_vs_{previous_year}'] = df.apply(
            lambda row: (row[current_year] - row[previous_year]) / row[previous_year] 
            if row[previous_year] != 0 else 0, axis=1
        )
    return df

def create_yearly_ratios(df, base_features):
    """
    Create ratio features between consecutive years.
    """
    for i in range(1, len(base_features)):
        current_year = base_features[i]
        previous_year = base_features[i-1]
        df[f'feat_ratio_{current_year}_to_{previous_year}'] = df.apply(
            lambda row: row[current_year] / row[previous_year] 
            if row[previous_year] != 0 else 0, axis=1
        )
    return df

def create_interaction_features(df, base_features):
    """
    Create interaction features between different years.
    """
    for i in range(len(base_features)):
        for j in range(i+1, len(base_features)):
            year1 = base_features[i]
            year2 = base_features[j]
            df[f'feat_interaction_{year1}_{year2}'] = df[year1] * df[year2]
    return df

def perform_feature_engineering(df, include_extra_features, base_features):
    """
    Perform feature engineering on the DataFrame if enabled.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        include_extra_features (bool): Flag to include/exclude extra features.
        base_features (list): List of base feature column names.
        
    Returns:
        pd.DataFrame: DataFrame with engineered features added.
    """
    if include_extra_features:
        print("Performing feature engineering...")
        
        # 1. Handle missing values (only numeric columns)
        df = handle_missing_values(df)
        
        # 2. Create statistical summary features
        df = create_statistical_features(df, base_features)
        
        # 3. Create year-over-year changes
        df = create_year_over_year_changes(df, base_features)
        
        # 4. Create growth rates
        df = create_growth_rates(df, base_features)
        
        # 5. Create yearly ratios
        df = create_yearly_ratios(df, base_features)
        
        # 6. Create interaction features
        df = create_interaction_features(df, base_features)
        
        # 7. (Optional) Add domain-specific features here
        # Example:
        # df = create_domain_specific_features(df)
        
        print("Feature engineering completed.")
    else:
        print("Skipping feature engineering.")
    return df


# ----------------------------------------------------
# 6) Automated Feature Selection Function
# ----------------------------------------------------
def get_feature_columns(df, include_extra_features):
    """
    Dynamically generate the list of feature columns based on whether extra features are included.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing all features.
        include_extra_features (bool): Flag to include/exclude extra features.
    
    Returns:
        list: List of feature column names.
    """
    if include_extra_features:
        # Select all columns that start with 'feat_' in addition to base features
        extra_features = [col for col in df.columns if col.startswith('feat_') and df[col].dtype in [np.float64, np.int64]]
        feature_columns = BASE_FEATURES + extra_features
        print(f"Extra features included: {extra_features}")
    else:
        # Use only base features
        feature_columns = BASE_FEATURES
        print("Only base features are included.")
    return feature_columns


# ----------------------------------------------------
# 7) Evaluation and Plotting
# ----------------------------------------------------
def evaluate_and_save_results(file_name, model_type, y_test, y_pred, scaler_y, 
                              predictions_folder, metrics, plot=True):
    """
    Evaluate model performance, log R² and MedAE, 
    and produce scatter + residual plots.
    """
    # 1) Inverse-transform predictions and test values to the original scale
    #    so we can compute metrics and residuals in real-world units.
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1))


    # 2) Compute metrics
    r2 = r2_score(y_test_inv, y_pred_inv)
    medae = median_absolute_error(y_test_inv, y_pred_inv)
    # rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    metrics.append({'File': file_name, 'Model': model_type, 'R²': r2, 'MedAE': medae})  #, 'RMSE': rmse})
    print(f"Metrics for {file_name} ({model_type}): R² = {r2:.4f}, MedAE = {medae:.4f}")

    if plot:
        # --------------------------------------------------
        # A) Scatter Plot: Actual vs. Predicted
        # --------------------------------------------------
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_inv, y_pred_inv, alpha=0.6, edgecolor='k')
        plt.plot([min(y_test_inv), max(y_test_inv)], 
                 [min(y_test_inv), max(y_test_inv)], 
                 'r--')
        plt.title(f'{file_name} - {model_type}\nR²={r2:.4f}, MedAE={medae:.4f}')  # , RMSE={rmse:.4f}')
        plt.xlabel('Actual DEC 2023')
        plt.ylabel('Predicted DEC 2023')
        plt.grid(True)
        plot_path = os.path.join(predictions_folder, f'scatter_{file_name}_{model_type}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Scatter plot saved to: {plot_path}")

        # --------------------------------------------------
        # B) Residuals: (y_test_inv - y_pred_inv)
        # --------------------------------------------------
        residuals = y_test_inv - y_pred_inv

        # (i) Residual Histogram
        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=100, alpha=0.7, edgecolor='k')
        plt.axvline(0, color='red', linestyle='--')
        plt.title(f'Residual Histogram ({file_name} - {model_type})')
        plt.xlabel('Residual (Actual - Predicted)')
        plt.ylabel('Count')
        plt.grid(True)
        hist_path = os.path.join(predictions_folder, f'residual_hist_{file_name}_{model_type}.png')
        plt.savefig(hist_path)
        plt.close()
        print(f"Residual histogram saved to: {hist_path}")

        # (ii) Residual Boxplot
        plt.figure(figsize=(8, 5))
        plt.boxplot(residuals, vert=True)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residual Boxplot ({file_name} - {model_type})')
        plt.ylabel('Residual')
        plt.grid(True)
        boxplot_path = os.path.join(predictions_folder, f'residual_box_{file_name}_{model_type}.png')
        plt.savefig(boxplot_path)
        plt.close()
        print(f"Residual boxplot saved to: {boxplot_path}")

# ----------------------------------------------------
# 8) Helper Function to Save Models and Scalers
# ----------------------------------------------------
def save_model_and_scalers(model, scaler_X, scaler_y, file_name, model_folder, model_type):
    """
    Save the trained model and the fitted scalers using joblib.
    """
    model_path = os.path.join(model_folder, f'{model_type}_model_{file_name}.pkl')
    scaler_X_path = os.path.join(model_folder, f'{model_type}_scaler_X_{file_name}.pkl')
    scaler_y_path = os.path.join(model_folder, f'{model_type}_scaler_y_{file_name}.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Scaler X saved to: {scaler_X_path}")
    print(f"Scaler y saved to: {scaler_y_path}")

# ----------------------------------------------------
# 9) Training Functions for Each Model
# ----------------------------------------------------
def train_linear_regression(X_train, y_train, X_test, y_test, file_name, model_folder, predictions_folder, metrics, scaler_X, scaler_y):
    """
    Train and evaluate a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train.ravel())
    save_model_and_scalers(model, scaler_X, scaler_y, file_name, model_folder, 'linear_regression')
    y_pred = model.predict(X_test)
    evaluate_and_save_results(file_name, 'linear_regression', y_test, y_pred, scaler_y, predictions_folder, metrics)

def train_knn(X_train, y_train, X_test, y_test, file_name, model_folder, predictions_folder, metrics, scaler_X, scaler_y):
    """
    Train and evaluate a K-Nearest Neighbors Regressor with hyperparameter tuning.
    """
    # Hyperparameter search for KNN
    param_grid = {
        'n_neighbors': [1, 2, 3, 4, 5],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # Manhattan or Euclidean
    }
    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train.ravel())
    model = grid_search.best_estimator_
    print(f"Best KNN Parameters for {file_name}: {grid_search.best_params_}")
    save_model_and_scalers(model, scaler_X, scaler_y, file_name, model_folder, 'knn')
    y_pred = model.predict(X_test)
    evaluate_and_save_results(file_name, 'knn', y_test, y_pred, scaler_y, predictions_folder, metrics)

def train_random_forest(X_train, y_train, X_test, y_test, file_name, model_folder, predictions_folder, metrics, scaler_X, scaler_y):
    """
    Train and evaluate a Random Forest Regressor with hyperparameter tuning using RandomizedSearchCV.
    Optimized for reduced training time.
    """
    # Define a reduced parameter grid
    param_dist = {
        'n_estimators': [100, 200],  # Reduced number of options
        'max_depth': [None, 10, 20],  # Removed 30 to reduce combinations
        'min_samples_split': [2, 5],  # Removed 10
        'min_samples_leaf': [1, 2]  # Removed 4
    }
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=10,  # Number of parameter settings sampled
        cv=3,  # Reduced number of cross-validation folds
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    print(f"Starting RandomizedSearchCV for {file_name}...")
    random_search.fit(X_train, y_train.ravel())
    model = random_search.best_estimator_
    print(f"Best Random Forest Parameters for {file_name}: {random_search.best_params_}")
    save_model_and_scalers(model, scaler_X, scaler_y, file_name, model_folder, 'random_forest')
    y_pred = model.predict(X_test)
    evaluate_and_save_results(file_name, 'random_forest', y_test, y_pred, scaler_y, predictions_folder, metrics)
    print(f"Random Forest model for {file_name} trained and saved successfully.\n")

def train_xgboost(X_train, y_train, X_test, y_test, file_name, model_folder, predictions_folder, metrics, scaler_X, scaler_y):
    """
    Train and evaluate an XGBoost Regressor with hyperparameter tuning.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    grid_search = GridSearchCV(
        XGBRegressor(random_state=42, objective='reg:squarederror'),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train.ravel())

    model = grid_search.best_estimator_
    print(f"Best XGBoost Parameters for {file_name}: {grid_search.best_params_}")
    save_model_and_scalers(model, scaler_X, scaler_y, file_name, model_folder, 'xgboost')
    y_pred = model.predict(X_test)
    evaluate_and_save_results(file_name, 'xgboost', y_test, y_pred, scaler_y, predictions_folder, metrics)

def train_lasso(X_train, y_train, X_test, y_test, file_name, model_folder, predictions_folder, metrics, scaler_X, scaler_y):
    """
    Train and evaluate a Lasso Regressor with hyperparameter tuning.
    """
    # Hyperparameter search for Lasso
    param_grid = {'alpha': [0.1, 1, 10, 100]}  # Regularization strength
    grid_search = GridSearchCV(Lasso(max_iter=10000), param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train.ravel())
    model = grid_search.best_estimator_
    print(f"Best Lasso Parameters for {file_name}: {grid_search.best_params_}")
    save_model_and_scalers(model, scaler_X, scaler_y, file_name, model_folder, 'lasso')
    y_pred = model.predict(X_test)
    evaluate_and_save_results(file_name, 'lasso', y_test, y_pred, scaler_y, predictions_folder, metrics)

def train_ridge(X_train, y_train, X_test, y_test, file_name, model_folder, predictions_folder, metrics, scaler_X, scaler_y):
    """
    Train and evaluate a Ridge Regressor with hyperparameter tuning.
    """
    # Hyperparameter search for Ridge
    param_grid = {'alpha': [0.1, 1, 10, 100]}  # Regularization strength
    grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train.ravel())
    model = grid_search.best_estimator_
    print(f"Best Ridge Parameters for {file_name}: {grid_search.best_params_}")
    save_model_and_scalers(model, scaler_X, scaler_y, file_name, model_folder, 'ridge')
    y_pred = model.predict(X_test)
    evaluate_and_save_results(file_name, 'ridge', y_test, y_pred, scaler_y, predictions_folder, metrics)

def train_svr(X_train, y_train, X_test, y_test, file_name, model_folder, predictions_folder, metrics, scaler_X, scaler_y):
    """
    Train and evaluate a Support Vector Regressor (SVR) with hyperparameter tuning.
    """
    # Define hyperparameter grid for SVR
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 1]
    }
    
    # Perform GridSearchCV for SVR
    grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train.ravel())  # SVR does not require reshaped y_train
    
    # Best model from GridSearchCV
    model = grid_search.best_estimator_
    print(f"Best SVR Parameters for {file_name}: {grid_search.best_params_}")
    
    # Save the model and scalers
    save_model_and_scalers(model, scaler_X, scaler_y, file_name, model_folder, 'svr')
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate and save results
    evaluate_and_save_results(file_name, 'svr', y_test, y_pred, scaler_y, predictions_folder, metrics)

def train_gpr(X_train, y_train, X_test, y_test, file_name, model_folder, predictions_folder, metrics, scaler_X, scaler_y):
    """
    Train Gaussian Process Regression (GPR) model, save the results and evaluate.
    """
    # Define kernel for GPR
    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))  # Combination of constant and RBF kernel
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

    # Fit the GPR model
    model.fit(X_train, y_train.ravel())

    # Save the trained model and scalers
    save_model_and_scalers(model, scaler_X, scaler_y, file_name, model_folder, 'gpr')

    # Predict on the test set
    y_pred, sigma = model.predict(X_test, return_std=True)

    # Evaluate and save results
    evaluate_and_save_results(file_name, 'gpr', y_test, y_pred, scaler_y, predictions_folder, metrics)

    # # Optionally, save uncertainty (sigma) for further analysis
    # uncertainty_path = os.path.join(predictions_folder, f'uncertainty_{file_name}_gpr.csv')
    # pd.DataFrame({'Prediction': y_pred.ravel(), 'Uncertainty': sigma}).to_csv(uncertainty_path, index=False)
    # print(f"Uncertainty saved to: {uncertainty_path}")

# ----------------------------------------------------
# 10) Main Training Loop with Automated Feature Selection and Saving Processed Data
# ----------------------------------------------------
# ----------------------------------------------------
# 10) Main Training Loop with Automated Feature Selection and Saving Processed Data
# ----------------------------------------------------
if __name__ == "__main__":
    # Get model folders based on the selected model_type
    model_folder, predictions_folder, summary_results_path = get_model_folders(model_type)
    metrics = []

    # Define the path for the test_data folder inside INPUT_FOLDER
    test_data_folder = os.path.join(INPUT_FOLDER, 'test_data')
    os.makedirs(test_data_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Initialize a list to collect all test data for a consolidated test.xlsx (optional)
    consolidated_test_data = []

    # Initialize a list to collect all processed DataFrames for merging
    merged_dfs = []

    for file_name in FILE_LIST:
        file_path = os.path.join(INPUT_FOLDER, f"{file_name}.xlsx")
        if not os.path.exists(file_path):
            print(f"File not found: {file_name}.xlsx")
            continue

        try:
            df = pd.read_excel(file_path)
            print(f"Successfully read file: {file_name}.xlsx")
        except Exception as e:
            print(f"Error reading {file_name}.xlsx: {e}")
            continue

        # Ensure required columns are present
        required = BASE_FEATURES + [TARGET_COLUMN]
        if not all(c in df.columns for c in required):
            print(f"Skipping {file_name} - required columns missing.")
            continue

        # Perform Feature Engineering if enabled
        df = perform_feature_engineering(df, INCLUDE_EXTRA_FEATURES, BASE_FEATURES)

        # Dynamically get feature columns
        feature_columns = get_feature_columns(df, INCLUDE_EXTRA_FEATURES)

        # Verify that feature_columns exist in the DataFrame
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            print(f"Skipping {file_name} - missing features after feature engineering: {missing_features}")
            continue

        # Ensure that feature_columns are numeric
        non_numeric_features = [col for col in feature_columns if df[col].dtype not in [np.float64, np.int64]]
        if non_numeric_features:
            print(f"Skipping {file_name} - non-numeric features detected: {non_numeric_features}")
            continue

        # Save the feature-engineered DataFrame to test_data/test_{file_name}.xlsx
        new_file_name = f'test_{file_name}.xlsx'
        new_file_path = os.path.join(test_data_folder, new_file_name)
        try:
            df.to_excel(new_file_path, index=False)
            print(f"Feature-engineered data saved to: {new_file_path}")
        except Exception as e:
            print(f"Error saving feature-engineered data to {new_file_path}: {e}")
            continue

        # Append the processed DataFrame to the list for merging
        merged_dfs.append(df.copy())  # Use .copy() to ensure independent DataFrames

        # Extract features and target
        X = df[feature_columns].values
        y = df[TARGET_COLUMN].values.reshape(-1, 1)

        # -------------------------
        # 1) Train/Test Split
        # -------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        # -------------------------
        # 2) Fit Scalers on TRAIN ONLY
        # -------------------------
        scaler_X = StandardScaler() #RobustScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)

        # -------------------------
        # 3) Transform TEST with same scaler
        # -------------------------
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)

        # -------------------------
        # 4) Pick which model to train
        # -------------------------
        if model_type == 'linear_regression':
            train_linear_regression(
                X_train_scaled, y_train_scaled,
                X_test_scaled, y_test_scaled,
                file_name, model_folder, predictions_folder, metrics,
                scaler_X, scaler_y
            )
        elif model_type == 'knn':
            train_knn(
                X_train_scaled, y_train_scaled,
                X_test_scaled, y_test_scaled,
                file_name, model_folder, predictions_folder, metrics,
                scaler_X, scaler_y
            )
        elif model_type == 'random_forest':
            train_random_forest(
                X_train_scaled, y_train_scaled,
                X_test_scaled, y_test_scaled,
                file_name, model_folder, predictions_folder, metrics,
                scaler_X, scaler_y
            )
        elif model_type == 'xgboost':
            train_xgboost(
                X_train_scaled, y_train_scaled,
                X_test_scaled, y_test_scaled,
                file_name, model_folder, predictions_folder, metrics,
                scaler_X, scaler_y
            )
        elif model_type == 'lasso':
            train_lasso(
                X_train_scaled, y_train_scaled,
                X_test_scaled, y_test_scaled,
                file_name, model_folder, predictions_folder, metrics,
                scaler_X, scaler_y
            )
        elif model_type == 'ridge':
            train_ridge(
                X_train_scaled, y_train_scaled,
                X_test_scaled, y_test_scaled,
                file_name, model_folder, predictions_folder, metrics,
                scaler_X, scaler_y
            )
        elif model_type == 'svr':
            train_svr(
                X_train_scaled, y_train_scaled,
                X_test_scaled, y_test_scaled,
                file_name, model_folder, predictions_folder, metrics,
                scaler_X, scaler_y
            )
        elif model_type == 'gpr':
            train_gpr(
                X_train_scaled, y_train_scaled,
                X_test_scaled, y_test_scaled,
                file_name, model_folder, predictions_folder, metrics,
                scaler_X, scaler_y
            )
        else:
            print(f"Unsupported model type: {model_type}. Skipping.")
            continue

        # -------------------------
        # 5) Collect Test Data for Consolidated test.xlsx (Optional)
        # -------------------------
        # Inverse transform the test data back to original scale for consolidation
        y_test_inv = scaler_y.inverse_transform(y_test_scaled)
        df_test_consolidated = pd.DataFrame(X_test_scaled, columns=feature_columns)
        df_test_consolidated[TARGET_COLUMN] = y_test_inv
        df_test_consolidated['Source_File'] = file_name  # Optional: Track the source file
        consolidated_test_data.append(df_test_consolidated)
        print(f"Test data from {file_name} added to consolidated test set.\n")

    # -------------------------
    # 6) Save Consolidated Test Data to test.xlsx (Optional)
    # -------------------------
    # Uncomment the following block if you want to save consolidated test data
    # if consolidated_test_data:
    #     try:
    #         consolidated_test_df = pd.concat(consolidated_test_data, ignore_index=True)
    #         test_output_path = os.path.join(OUTPUT_BASE_FOLDER, 'test.xlsx')
    #         consolidated_test_df.to_excel(test_output_path, index=False)
    #         print(f"All consolidated test data saved to: {test_output_path}")
    #     except Exception as e:
    #         print(f"Error saving consolidated test data: {e}")
    # else:
    #     print("No test data to save.")

    # -------------------------
    # 7) Save Metrics Summary
    # -------------------------
    if metrics:
        try:
            summary_df = pd.DataFrame(metrics)
            summary_df.to_csv(summary_results_path, index=False)
            print(f"\nSummary results saved to: {summary_results_path}")
        except Exception as e:
            print(f"Error saving summary results: {e}")
    else:
        print("\nNo metrics to save. Check if all files were processed correctly.")

    # -------------------------
    # 8) Merge All Feature-Engineered DataFrames into One Excel File
    # -------------------------
    if merged_dfs:
        try:
            # Concatenate all DataFrames in the list
            merged_features_df = pd.concat(merged_dfs, ignore_index=True)

            # Define the path for the merged Excel file
            merged_output_path = os.path.join(INPUT_FOLDER, 'test_data', 'merged_features.xlsx')

            # Save the merged DataFrame to Excel
            merged_features_df.to_excel(merged_output_path, index=False)
            print(f"All feature-engineered data merged and saved to: {merged_output_path}")
        except Exception as e:
            print(f"Error merging and saving feature-engineered data: {e}")
    else:
        print("No DataFrames to merge.")

    # Save metrics summary
    if metrics:
        try:
            summary_df = pd.DataFrame(metrics)
            summary_df.to_csv(summary_results_path, index=False)
            print(f"\nSummary results saved to: {summary_results_path}")
        except Exception as e:
            print(f"Error saving summary results: {e}")
    else:
        print("\nNo metrics to save. Check if all files were processed correctly.")