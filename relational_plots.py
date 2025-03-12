import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------
# 1) Adjust these paths/folders as needed
# ----------------------------------------------------
CURRENT_DIR = os.getcwd()
INPUT_FOLDER = os.path.join(CURRENT_DIR, 'data')
RELATIONS_FOLDER = os.path.join(CURRENT_DIR, 'relational_plots')

# Create the relations folder if it does not exist
os.makedirs(RELATIONS_FOLDER, exist_ok=True)

# File list containing the dataset names
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
# 2) Function to Read Files
# ----------------------------------------------------
def read_files(input_folder, file_list):
    data_frames = {}
    for file_name in file_list:
        file_path = os.path.join(input_folder, f"{file_name}.xlsx")
        if os.path.exists(file_path):
            data_frames[file_name] = pd.read_excel(file_path)
        else:
            print(f"[WARNING] File not found: {file_name}.xlsx")
    return data_frames

# ----------------------------------------------------
# 3) Function to Process Data
# ----------------------------------------------------
def process_data(df):
    required_columns = ['DEC 2019', 'DEC 2020', 'DEC 2021', 'DEC 2022', 'DEC 2023']
    for col in required_columns:
        if col not in df.columns:
            print(f"[WARNING] Missing column: {col}")
    # Drop rows where all required columns are NaN
    df = df.dropna(subset=required_columns, how='all')
    # Interpolate missing values
    df = df.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')
    return df

# ----------------------------------------------------
# 4) Function to Generate Plots
# ----------------------------------------------------
def generate_plots(data_frames):
    for file_name, df in data_frames.items():
        print(f"Generating plots for: {file_name}")
        features = ['DEC 2019', 'DEC 2020', 'DEC 2021', 'DEC 2022']
        target = 'DEC 2023'
        
        # Create a subfolder for each dataset within the relations folder
        file_relations_folder = os.path.join(RELATIONS_FOLDER, file_name)
        os.makedirs(file_relations_folder, exist_ok=True)
        
        # 1. Scatter Plots
        for feature in features:
            plt.figure(figsize=(6, 4))
            plt.scatter(df[feature], df[target], alpha=0.6, edgecolor='k')
            plt.title(f'{feature} vs {target} ({file_name})')
            plt.xlabel(feature)
            plt.ylabel(target)
            plt.grid(True)
            scatter_plot_path = os.path.join(file_relations_folder, f'{feature}_scatter.png')
            plt.savefig(scatter_plot_path)
            plt.close()
            print(f"Saved scatter plot: {scatter_plot_path}")

        # # 2. Pairplot (all columns together)
        # pairplot_data = df[features + [target]].copy()

        # # Downsample if too large
        # if len(pairplot_data) > 5000:
        #         pairplot_data = pairplot_data.sample(n=5000, random_state=42)

        # pairplot_path = os.path.join(file_relations_folder, 'pairplot.png')
        # sns.pairplot(
        #     pairplot_data, 
        #     corner=True,
        #     plot_kws={'s': 2, 'alpha': 0.5},  # smaller markers, transparency
        #     diag_kind="hist",                 # hist instead of kde (often faster)
        #     )
        # plt.suptitle(f'Pairplot ({file_name})', y=1.02)
        # plt.savefig(pairplot_path)
        # plt.close()
        # print(f"Saved pairplot: {pairplot_path}")


        
        
        # 2. Correlation Heatmap
        plt.figure(figsize=(8, 6))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title(f'Correlation Heatmap ({file_name})')
        heatmap_path = os.path.join(file_relations_folder, 'correlation_heatmap.png')
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Saved correlation heatmap: {heatmap_path}")

        # # 3. Residual Plot (example with Linear Regression)
        # X = df[features].values
        # y = df[target].values
        # model = LinearRegression()
        # model.fit(X, y)
        # y_pred = model.predict(X)
        # residuals = y - y_pred
        
        # plt.figure(figsize=(6, 4))
        # plt.scatter(y_pred, residuals, alpha=0.6, edgecolor='k')
        # plt.axhline(0, color='red', linestyle='--')
        # plt.title(f'Residual Plot ({file_name})')
        # plt.xlabel('Predicted DEC2023')
        # plt.ylabel('Residuals')
        # plt.grid(True)
        # residual_plot_path = os.path.join(file_relations_folder, 'residual_plot.png')
        # plt.savefig(residual_plot_path)
        # plt.close()
        # print(f"Saved residual plot: {residual_plot_path}")
       
        # 3. Histograms for each DEC column (to see distribution)
        all_dec_columns = features + [target]
        for col in all_dec_columns:
            plt.figure(figsize=(6,4))
            sns.histplot(df[col], kde=False, bins=100)
            plt.title(f'Histogram for {col} ({file_name})')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.grid(True)
            hist_path = os.path.join(file_relations_folder, f'histogram_{col}.png')
            plt.savefig(hist_path)
            plt.close()
            print(f"Saved histogram for {col}: {hist_path}")

        # 4. Boxplots for each DEC column (to check for outliers)
        for col in all_dec_columns:
            plt.figure(figsize=(6,4))
            sns.boxplot(y=df[col], orient='v')
            plt.title(f'Boxplot for {col} ({file_name})')
            plt.ylabel(col)
            plt.grid(True)
            boxplot_path = os.path.join(file_relations_folder, f'boxplot_{col}.png')
            plt.savefig(boxplot_path)
            plt.close()
            print(f"Saved boxplot for {col}: {boxplot_path}")
# ----------------------------------------------------
# 5) Main Execution
# ----------------------------------------------------
if __name__ == "__main__":
    # Read files
    data_frames = read_files(INPUT_FOLDER, FILE_LIST)

    # Process data and generate plots
    for file_name, df in data_frames.items():
        df = process_data(df)  # Process each DataFrame
        generate_plots({file_name: df})  # Generate plots for each DataFrame
