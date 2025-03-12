# """
# Created on Wed Jan  8 09:27:32 2025

# @author: metinvural
# """

# import pandas as pd

# # Load the Excel file
# file_path = "data/0_dataset.xlsx" # Replace with your actual file path
# df = pd.read_excel(file_path)

# # Display the first few rows to ensure it loaded correctly
# print(df.head())

# # Extract unique currencies
# unique_currencies = df["CURRENCY"].unique()

# # Print the unique currencies
# print("Unique currencies:", unique_currencies)

# """
# Unique currencies: ['GBP' 'NGN' 'NOK' 'SEK' 'USD' 'DKK' 'PLN' 'EUR' 'KRW' 'HKD' 'SGD' 'INR'
#  'JPY' 'IDR' 'ZAR' 'AUD' 'CAD' 'PKR' 'ZWG' 'THB' 'BRL' 'EGP' 'DEM' 'TWD'
#  'CNY' 'BHD' 'VND' 'NZD' 'PHP' 'SAR' 'CHF' 'RON' 'MXN' 'ILS' 'MYR' 'AED'
#  'ROL' 'AMD' 'GRD' 'UYU' 'CZK' 'PEN' 'ISK' 'KWD' 'OMR' 'MAD' 'HUF' 'JMD'
#  'RUB' 'BGN' 'TRY' 'KZT' 'CLP' 'EEK']
# """



import pandas as pd
import numpy as np
import os

# Define the target currency
target_currency = "EUR"  # Change this to "USD" if needed

# Define the file path dynamically based on the target currency
input_folder = "./data"
output_folder = f"./data_{target_currency.lower()}"
os.makedirs(output_folder, exist_ok=True)

file_path = f"{input_folder}/dataset-120FA5A4-D24E-11EF-9488-97B158A30D87.xlsx"
output_file = f"{output_folder}/converted_dataset_to_{target_currency.lower()}.xlsx"

# Define the years and corresponding columns
years = [2019, 2020, 2021, 2022, 2023]
columns = [f"DEC {year}" for year in years]


exchange_rates = {
    'GBP': {'EUR': [1.17, 1.15, 1.18, 1.19, 1.21], 'USD': [1.31, 1.30, 1.33, 1.34, 1.36]},
    'NGN': {'EUR': [0.0025, 0.0024, 0.0026, 0.0028, 0.003], 'USD': [0.0028, 0.0027, 0.0029, 0.003, 0.0032]},
    'NOK': {'EUR': [0.10, 0.095, 0.096, 0.098, 0.097], 'USD': [0.11, 0.10, 0.11, 0.12, 0.11]},
    'SEK': {'EUR': [0.093, 0.092, 0.0925, 0.0935, 0.094], 'USD': [0.10, 0.098, 0.099, 0.101, 0.102]},
    'USD': {'EUR': [0.89, 0.85, 0.87, 0.90, 0.91], 'USD': [1.0, 1.0, 1.0, 1.0, 1.0]},
    'DKK': {'EUR': [0.134, 0.133, 0.1325, 0.135, 0.136], 'USD': [0.15, 0.152, 0.153, 0.154, 0.155]},
    'PLN': {'EUR': [0.23, 0.232, 0.231, 0.234, 0.236], 'USD': [0.25, 0.26, 0.255, 0.257, 0.259]},
    'EUR': {'EUR': [1.0, 1.0, 1.0, 1.0, 1.0], 'USD': [1.12, 1.18, 1.16, 1.11, 1.10]},
    'KRW': {'EUR': [0.00077, 0.00076, 0.00078, 0.00079, 0.0008], 'USD': [0.00085, 0.00083, 0.00086, 0.00087, 0.00088]},
    'HKD': {'EUR': [0.114, 0.113, 0.115, 0.116, 0.117], 'USD': [0.13, 0.132, 0.134, 0.135, 0.136]},
    'SGD': {'EUR': [0.66, 0.65, 0.67, 0.68, 0.69], 'USD': [0.74, 0.73, 0.75, 0.76, 0.77]},
    'INR': {'EUR': [0.0125, 0.0124, 0.0126, 0.0127, 0.0128], 'USD': [0.014, 0.0139, 0.0141, 0.0142, 0.0143]},
    'JPY': {'EUR': [0.0081, 0.0080, 0.0082, 0.0083, 0.0084], 'USD': [0.009, 0.0089, 0.0091, 0.0092, 0.0093]},
    'IDR': {'EUR': [0.00006, 0.00005, 0.00007, 0.00008, 0.00009], 'USD': [0.00007, 0.00006, 0.00008, 0.00009, 0.0001]},
    'ZAR': {'EUR': [0.055, 0.054, 0.056, 0.057, 0.058], 'USD': [0.062, 0.061, 0.063, 0.064, 0.065]},
    'AUD': {'EUR': [0.62, 0.63, 0.64, 0.65, 0.66], 'USD': [0.69, 0.7, 0.71, 0.72, 0.73]},
    'CAD': {'EUR': [0.65, 0.64, 0.66, 0.67, 0.68], 'USD': [0.74, 0.73, 0.75, 0.76, 0.77]},
    'PKR': {'EUR': [0.0057, 0.0056, 0.0058, 0.0059, 0.006], 'USD': [0.0065, 0.0064, 0.0066, 0.0067, 0.0068]},
    'ZWG': {'EUR': [0.00003, 0.00003, 0.00004, 0.00004, 0.00005], 'USD': [0.00004, 0.00004, 0.00005, 0.00005, 0.00006]},
    'THB': {'EUR': [0.029, 0.028, 0.030, 0.031, 0.032], 'USD': [0.034, 0.033, 0.035, 0.036, 0.037]},
    'BRL': {'EUR': [0.18, 0.175, 0.19, 0.195, 0.20], 'USD': [0.21, 0.205, 0.22, 0.225, 0.23]},
    'EGP': {'EUR': [0.056, 0.055, 0.058, 0.059, 0.06], 'USD': [0.064, 0.063, 0.067, 0.068, 0.069]},
    'DEM': {'EUR': [1.96, 1.95, 1.94, 1.97, 1.98], 'USD': [2.10, 2.08, 2.09, 2.12, 2.13]},
    'TWD': {'EUR': [0.029, 0.028, 0.030, 0.031, 0.032], 'USD': [0.033, 0.032, 0.034, 0.035, 0.036]},
    'CNY': {'EUR': [0.13, 0.14, 0.15, 0.16, 0.17], 'USD': [0.15, 0.16, 0.17, 0.18, 0.19]},
    'BHD': {'EUR': [2.35, 2.34, 2.36, 2.37, 2.38], 'USD': [2.50, 2.49, 2.52, 2.53, 2.54]},
    'VND': {'EUR': [0.00004, 0.00003, 0.00005, 0.00006, 0.00007], 'USD': [0.00005, 0.00004, 0.00006, 0.00007, 0.00008]},
    'NZD': {'EUR': [0.58, 0.59, 0.60, 0.61, 0.62], 'USD': [0.65, 0.66, 0.67, 0.68, 0.69]},
    'PHP': {'EUR': [0.017, 0.0165, 0.0175, 0.018, 0.0185], 'USD': [0.019, 0.0185, 0.0195, 0.02, 0.0205]},
    'SAR': {'EUR': [0.24, 0.245, 0.25, 0.255, 0.26], 'USD': [0.27, 0.275, 0.28, 0.285, 0.29]},
    'CHF': {'EUR': [0.92, 0.91, 0.93, 0.94, 0.95], 'USD': [1.0, 0.99, 1.01, 1.02, 1.03]},
    'RON': {'EUR': [0.21, 0.22, 0.23, 0.24, 0.25], 'USD': [0.23, 0.24, 0.25, 0.26, 0.27]},
    'MXN': {'EUR': [0.042, 0.041, 0.043, 0.044, 0.045], 'USD': [0.047, 0.046, 0.048, 0.049, 0.05]},
    'ILS': {'EUR': [0.25, 0.26, 0.27, 0.28, 0.29], 'USD': [0.28, 0.29, 0.30, 0.31, 0.32]},
    'MYR': {'EUR': [0.22, 0.23, 0.24, 0.25, 0.26], 'USD': [0.25, 0.26, 0.27, 0.28, 0.29]},
    'AED': {'EUR': [0.24, 0.25, 0.26, 0.27, 0.28], 'USD': [0.27, 0.28, 0.29, 0.30, 0.31]},
    'ROL': {'EUR': [0.21, 0.22, 0.23, 0.24, 0.25], 'USD': [0.24, 0.25, 0.26, 0.27, 0.28]},
    'AMD': {'EUR': [0.0021, 0.0022, 0.0023, 0.0024, 0.0025], 'USD': [0.0024, 0.0025, 0.0026, 0.0027, 0.0028]},
    'GRD': {'EUR': [0.003, 0.0031, 0.0032, 0.0033, 0.0034], 'USD': [0.0034, 0.0035, 0.0036, 0.0037, 0.0038]},
    'UYU': {'EUR': [0.024, 0.025, 0.026, 0.027, 0.028], 'USD': [0.027, 0.028, 0.029, 0.03, 0.031]},
    'CZK': {'EUR': [0.041, 0.042, 0.043, 0.044, 0.045], 'USD': [0.046, 0.047, 0.048, 0.049, 0.05]},
    'PEN': {'EUR': [0.23, 0.24, 0.25, 0.26, 0.27], 'USD': [0.26, 0.27, 0.28, 0.29, 0.3]},
    'ISK': {'EUR': [0.0075, 0.0076, 0.0077, 0.0078, 0.0079], 'USD': [0.0085, 0.0086, 0.0087, 0.0088, 0.0089]},
    'KWD': {'EUR': [2.96, 2.97, 2.98, 2.99, 3.0], 'USD': [3.24, 3.25, 3.26, 3.27, 3.28]},
    'OMR': {'EUR': [2.38, 2.39, 2.4, 2.41, 2.42], 'USD': [2.55, 2.56, 2.57, 2.58, 2.59]},
    'MAD': {'EUR': [0.092, 0.093, 0.094, 0.095, 0.096], 'USD': [0.105, 0.106, 0.107, 0.108, 0.109]},
    'HUF': {'EUR': [0.0027, 0.0028, 0.0029, 0.003, 0.0031], 'USD': [0.0031, 0.0032, 0.0033, 0.0034, 0.0035]},
    'JMD': {'EUR': [0.0067, 0.0068, 0.0069, 0.007, 0.0071], 'USD': [0.0075, 0.0076, 0.0077, 0.0078, 0.0079]},
    'RUB': {'EUR': [0.013, 0.012, 0.0135, 0.014, 0.0145], 'USD': [0.015, 0.0145, 0.016, 0.0165, 0.017]},
    'BGN': {'EUR': [0.51, 0.52, 0.53, 0.54, 0.55], 'USD': [0.57, 0.58, 0.59, 0.6, 0.61]},
    'TRY': {'EUR': [0.18, 0.175, 0.19, 0.195, 0.2], 'USD': [0.21, 0.205, 0.22, 0.225, 0.23]},
    'KZT': {'EUR': [0.0023, 0.0024, 0.0025, 0.0026, 0.0027], 'USD': [0.0026, 0.0027, 0.0028, 0.0029, 0.003]},
    'CLP': {'EUR': [0.0012, 0.0013, 0.0014, 0.0015, 0.0016], 'USD': [0.0013, 0.0014, 0.0015, 0.0016, 0.0017]},
    'EEK': {'EUR': [0.06, 0.059, 0.058, 0.057, 0.056], 'USD': [0.068, 0.067, 0.066, 0.065, 0.064]},
}


    # Read the dataset
# df = pd.read_excel(file_path)
df = pd.read_excel(file_path, keep_default_na=True)  # Use this to preserve NaNs

df = df.dropna(subset=['CURRENCY'])
df = df[df['CURRENCY'] != 'UNKNOWN']

# Extract relevant currencies from the dataset
currencies_in_data = df['CURRENCY'].unique()
missing_currencies = [currency for currency in currencies_in_data if currency not in exchange_rates]

# Check for missing currencies
if missing_currencies:
    raise ValueError(f"The following currencies are missing exchange rates: {missing_currencies}")

# Ensure the columns are numeric
df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')

# Handle NaN values (choose one option):
# df[columns] = df[columns].fillna(0)  # Replace NaN with 0
# Alternatively, drop rows with NaN values in the specified columns:
# df = df.dropna(subset=columns)

# # Function to convert values
# def convert_value(row, year_index, target_currency):
#     currency = row['CURRENCY']
#     rate = exchange_rates[currency][target_currency][year_index]
#     value = row[columns[year_index]]
#     if not isinstance(value, (int, float)):
#         print(f"Non-numeric value found: {value} in column {columns[year_index]} for row: {row}")
#         return None  # Skip problematic rows or handle as needed
#     return value * rate
# Function to convert values
def convert_value(row, year_index, target_currency):
    currency = row['CURRENCY']
    rate = exchange_rates[currency][target_currency][year_index]
    value = row[columns[year_index]]
    if pd.isna(value):  # Check for NaN
        return np.nan  # Preserve NaN explicitly
    return value * rate

# Convert values for each year and update the dataset in place
for year_index, column in enumerate(columns):
    df[column] = df.apply(lambda row: convert_value(row, year_index, target_currency), axis=1)

# Update the CURRENCY column to reflect the target currency
df['CURRENCY'] = target_currency

# Save the updated dataset
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, na_rep='')  # na_rep='' ensures NaNs are preserved as blank cells
print(f"Dataset successfully converted to {target_currency} and saved to '{output_file}'.")

# Create a test_data folder inside the output folder
test_data_folder = os.path.join(output_folder, "test_data")
os.makedirs(test_data_folder, exist_ok=True)
# Define the path for the test file
test_file_path = os.path.join(test_data_folder, "test.xlsx")
# Save a copy of the converted dataset as test.xlsx
with pd.ExcelWriter(test_file_path, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, na_rep='')  # Preserve NaNs as blank cells
print(f"A copy of the converted dataset has been saved to '{test_file_path}'.")


# Split the dataset by SECTION and save separate files
section_file_mapping = {
    'Capital Expenditures': '02_capital_expenditures.xlsx',
    'Cash & Short-Term Investments': '03_cash_short_term_inv.xlsx',
    'EBIT': '04_EBIT.xlsx',
    'EBITDA': '05_EBITDA.xlsx',
    'Free Cash Flow': '06_Free_cash_flow.xlsx',
    'Gross Income': '07_Gross_income.xlsx',
    'Net Debt': '08_Net Debt.xlsx',
    'Net Financing Cash Flow': '09_Net_Fin_Cash_Flow.xlsx',
    'Net Income': '10_Net_income.xlsx',
    'Net Investing Cash Flow': '11_Net_Inv_cash_flow.xlsx',
    'Net Operating Cash Flow': '12_Net_Op_Cash_Flow.xlsx',
    'Sales': '13_Sales.xlsx',
    'Total Assets': '14_Total_assets.xlsx',
    'Total Debt': '15_Total_debt.xlsx',
    'Total Liabilities': '16_Total_Liabilities.xlsx',
    'Total Shareholders\' Equity': '17_Total_sharehold_eq.xlsx',
}

# Split the dataset by SECTION and save separate files
for section, file_name in section_file_mapping.items():
    section_data = df[
        (df['SECTION'] == section) & 
        df[columns].notna().all(axis=1)  # Filter rows where all DEC columns have valid (non-NaN) data
    ]
    section_data = pd.concat([section_data, df[['NAME']]], axis=1)  # Include the NAME column])

    output_path = os.path.join(output_folder, file_name)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        section_data.to_excel(writer, index=False, na_rep='')  # Preserve NaNs as blank cells
    print(f"Saved data for '{section}' to '{output_path}'.")


# a= df = df.head(10)