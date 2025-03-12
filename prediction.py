import os
import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.impute import SimpleImputer  # Optional: If using imputation

# ------------------------------------------------------------------
# 1) Configuration Options
# ------------------------------------------------------------------

# Specify the target currency. Options: "EUR", "USD", "generic", etc.
TARGET_CURRENCY = "EUR"  # Change as needed

# Single flag to control both input and output folders
USE_DEFAULT_FOLDERS = False  # True = 'data' & 'output', False = 'data_eur' & 'output_eur'

# ------------------------------------------------------------------
# 2) Adjust Paths/Folders Dynamically
# ------------------------------------------------------------------

CURRENT_DIR = os.getcwd()

# Determine input and output folders based on the single flag
if USE_DEFAULT_FOLDERS:
    INPUT_FOLDER_BASE = os.path.join(CURRENT_DIR, "data")
    OUTPUT_BASE_FOLDER = os.path.join(CURRENT_DIR, "output")
else:
    # Handle special case for 'generic' if needed
    if TARGET_CURRENCY.lower() == "generic":
        INPUT_FOLDER_BASE = os.path.join(CURRENT_DIR, "data")
        OUTPUT_BASE_FOLDER = os.path.join(CURRENT_DIR, "output")
    else:
        INPUT_FOLDER_BASE = os.path.join(CURRENT_DIR, f"data_{TARGET_CURRENCY.lower()}")
        OUTPUT_BASE_FOLDER = os.path.join(CURRENT_DIR, f"output_{TARGET_CURRENCY.lower()}")

print("Using INPUT_FOLDER_BASE:", INPUT_FOLDER_BASE)
print("Using OUTPUT_BASE_FOLDER:", OUTPUT_BASE_FOLDER)

# ------------------------------------------------------------------
# 3) Define Model-Specific Folders and Paths
# ------------------------------------------------------------------

def get_model_folders(model_type):
    """
    Retrieves the paths for the model folder and summary results based on the model type.
    
    Parameters:
        model_type (str): The type of the model (e.g., 'ridge', 'knn').
    
    Returns:
        tuple: (model_folder, base_folder, summary_results_path)
    """
    base_folder = os.path.join(OUTPUT_BASE_FOLDER, model_type)
    model_folder = os.path.join(base_folder, 'models')
    summary_results_path = os.path.join(base_folder, f'summary_results_{model_type}.csv')
    os.makedirs(model_folder, exist_ok=True)
    return model_folder, base_folder, summary_results_path

# ------------------------------------------------------------------
# 4) Specify Model Type and Get Folders
# ------------------------------------------------------------------
MODEL_TYPE = 'knn'  # Options: 'knn', 'linear_regression', 'random_forest', 'xgboost', 'lasso', 'ridge', 'svr'

model_folder, base_folder, summary_results_path = get_model_folders(MODEL_TYPE)

# Define where predictions will be saved
OUTPUT_FILE_PATH = os.path.join(base_folder, 'predictions_test_results.xlsx')
print(f"Predictions will be saved to: {OUTPUT_FILE_PATH}")

# ------------------------------------------------------------------
# 5) Load and Prepare Data
# ------------------------------------------------------------------

# Define the input file path dynamically to point to merged_features.xlsx
INPUT_FILE_PATH = os.path.join(INPUT_FOLDER_BASE, 'test_data_1', 'merged_features.xlsx')

# Print input file path for debugging
print(f"Using input file: {INPUT_FILE_PATH}")
print(f"Using output file: {OUTPUT_FILE_PATH}")

# Check if the input file exists
if not os.path.exists(INPUT_FILE_PATH):
    raise FileNotFoundError(f"Input file not found at {INPUT_FILE_PATH}")

# Load the data
data = pd.read_excel(INPUT_FILE_PATH)

# Define base and target features
base_input_columns = ['DEC 2019', 'DEC 2020', 'DEC 2021', 'DEC 2022']
target_column = 'DEC 2023'

# # Example: Handle missing values if necessary
# imputer = SimpleImputer(strategy='mean')  # or any other strategy
# data[base_input_columns] = imputer.fit_transform(data[base_input_columns])


# Map SECTION to the correct file name used in training
section_to_model_map = {
    "Capital Expenditures": "02_capital_expenditures",
    "Cash & Short-Term Investments": "03_cash_short_term_inv",
    "EBIT": "04_EBIT",
    "EBITDA": "05_EBITDA",
    "Free Cash Flow": "06_Free_cash_flow",
    "Gross Income": "07_Gross_income",
    "Net Debt": "08_Net Debt",
    "Net Financing Cash Flow": "09_Net_Fin_Cash_Flow",
    "Net Income": "10_Net_income",
    "Net Investing Cash Flow": "11_Net_Inv_cash_flow",
    "Net Operating Cash Flow": "12_Net_Op_Cash_Flow",
    "Sales": "13_Sales",
    "Total Assets": "14_Total_assets",
    "Total Debt": "15_Total_debt",
    "Total Liabilities": "16_Total_Liabilities",
    "Total Shareholders' Equity": "17_Total_sharehold_eq"
}

# ------------------------------------------------------------------
# 4) Function to Automatically Detect Extra Features
# ------------------------------------------------------------------
def get_extra_features(df, base_features, target_column, exclude_columns=None):
    """
    Automatically identify extra feature columns in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing all features.
        base_features (list): List of base feature column names.
        target_column (str): The target column name.
        exclude_columns (list, optional): Additional columns to exclude (e.g., identifiers).
    
    Returns:
        list: List of extra feature column names.
    """
    if exclude_columns is None:
        exclude_columns = ['SECTION']  # Add any other identifier columns here
    
    all_columns = df.columns.tolist()
    # Extra features are all columns not in base_features, target_column, or exclude_columns
    extra_features = [col for col in all_columns 
                      if col not in base_features 
                      and col != target_column 
                      and col not in exclude_columns]
    return extra_features

# ------------------------------------------------------------------
# 5) Load & Apply the Model
# ------------------------------------------------------------------
def apply_model(section_value, input_data, input_columns):
    """
    Look up the model/scalers by section_value,
    transform input_data, predict, and invert the scaling.
    
    Parameters:
        section_value (str): The SECTION value to determine which model to use.
        input_data (np.ndarray): The input feature data for prediction.
        input_columns (list): List of input feature column names.
    
    Returns:
        np.ndarray or None: The predicted target values or None if an error occurs.
    """
    model_name = section_to_model_map.get(section_value)
    if not model_name:
        print(f"[WARNING] SECTION not found in map: {section_value}")
        return None

    # .pkl file paths
    model_path = os.path.join(model_folder, f'{MODEL_TYPE}_model_{model_name}.pkl')
    scaler_X_path = os.path.join(model_folder, f'{MODEL_TYPE}_scaler_X_{model_name}.pkl')
    scaler_y_path = os.path.join(model_folder, f'{MODEL_TYPE}_scaler_y_{model_name}.pkl')
    
    # Check existence
    missing_files = []
    for f in [model_path, scaler_X_path, scaler_y_path]:
        if not os.path.exists(f):
            missing_files.append(f)

    if missing_files:
        print(f"[ERROR] Missing model/scaler files for SECTION '{section_value}'")
        for m in missing_files:
            print(f"   -> {m}")
        return None

    # Load model & scalers
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model from {model_path}: {e}")
        return None

    try:
        scaler_X = joblib.load(scaler_X_path)
    except Exception as e:
        print(f"[ERROR] Failed to load scaler_X from {scaler_X_path}: {e}")
        return None

    try:
        scaler_y = joblib.load(scaler_y_path)
    except Exception as e:
        print(f"[ERROR] Failed to load scaler_y from {scaler_y_path}: {e}")
        return None

    # Scale the input
    try:
        X_scaled = scaler_X.transform(input_data)
    except Exception as e:
        print(f"[ERROR] Failed to scale input data: {e}")
        return None

    # Predict
    try:
        y_scaled_pred = model.predict(X_scaled)
    except Exception as e:
        print(f"[ERROR] Failed to make predictions: {e}")
        return None

    # Invert scaling
    try:
        y_pred = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1))
    except Exception as e:
        print(f"[ERROR] Failed to inverse transform predictions: {e}")
        return None

    return y_pred

# ------------------------------------------------------------------
# 6) Read Test Data & Produce Predictions
# ------------------------------------------------------------------
# Read the feature-engineered test data
df_test = pd.read_excel(INPUT_FILE_PATH)

# Replace '-' with NaN in the target column if it exists
if target_column in df_test.columns:
    df_test[target_column] = df_test[target_column].replace('-', np.nan)

# Automatically identify extra features
extra_features = get_extra_features(df_test, base_input_columns, target_column)

# Combine base and extra features for input
input_columns = base_input_columns + extra_features

# Print identified extra features for verification
print(f"Identified Extra Features: {extra_features}")

# Check if all required input columns are present
required_cols = ['SECTION'] + input_columns
missing = [c for c in required_cols if c not in df_test.columns]
if missing:
    raise ValueError(f"[ERROR] The test file is missing required columns: {missing}")

# Create a copy of the test DataFrame for results
df_results = df_test.copy()

# Handle missing values in input columns by dropping incomplete rows
df_results = df_results.dropna(subset=input_columns).reset_index(drop=True)

# Optional: Impute missing values instead of dropping (uncomment if needed)
# imputer = SimpleImputer(strategy='mean')
# df_results[input_columns] = imputer.fit_transform(df_results[input_columns])

# Initialize tqdm for progress visualization
sections = df_results['SECTION'].unique()
for section_value in tqdm(sections, desc="Processing sections"):
    section_data = df_results[df_results['SECTION'] == section_value]

    # Identify rows that have complete input
    valid_mask = section_data[input_columns].notna().all(axis=1)
    valid_indices = section_data[valid_mask].index

    if valid_indices.empty:
        continue

    # Prepare data for prediction
    input_data = section_data.loc[valid_indices, input_columns].values

    # Predict
    predictions = apply_model(section_value, input_data, input_columns)

    if predictions is not None:
        # Assign predictions to the target column in the results DataFrame
        df_results.loc[valid_indices, target_column] = predictions.flatten()

# ------------------------------------------------------------------
# 7) Save Final Inference Predictions
# ------------------------------------------------------------------
os.makedirs(base_folder, exist_ok=True)
output_path = os.path.join(base_folder, f"{MODEL_TYPE}_predictions_test_results.xlsx")
df_results.to_excel(output_path, index=False)
print(f"\nInference predictions saved to: {output_path}")
