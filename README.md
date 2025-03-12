# Financial Forecasting

## Overview
This project uses **KNN, Lasso, Ridge, Linear Regression** to predict financial metrics (e.g., `DEC 2023`) based on historical data (`DEC 2019–2022`). It handles cases where some rows already have `DEC 2023` values by:
- Using available values for training the model.
- Predicting missing `DEC 2023` values.
- Combining the results into a complete dataset.

The project also evaluates the model's performance using metrics like **R² Score** and **Median Absolute Error (MedAE)**.

---

## Mathematical Formulation

### Linear Regression Equation
The predicted target value ($` \hat{y} `$) is calculated as:
```math
\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
```
Where:
- $` x_1, x_2, \dots, x_n `$: Input features (historical data).
- $` w_1, w_2, \dots, w_n `$: Coefficients (weights) learned during training.
- $` b `$: Intercept (bias term).


---

## Workflow

### 1. Input Data
The input dataset consists of financial metrics for multiple companies over several years. Example:

| Section      | DEC 2019 | DEC 2020 | DEC 2021 | DEC 2022 | DEC 2023 |
|--------------|----------|----------|----------|----------|----------|
| Sales        | 1000     | 1100     | 1200     | 1300     | 1400     |
| Sales        | 2000     | 2100     | 2200     | 2300     |       |
| Sales        | 1500     | 1600     | 1700     | 1800     | 1900     |

### 2. Handling Missing `DEC 2023` Values
- Rows with available `DEC 2023` values are used to **train the model**.
- Rows with missing `DEC 2023` values are used for **prediction**.

### 3. Training
- The model learns a linear relationship between historical values (`DEC 2019–2022`) and the target (`DEC 2023`).
- A single model is trained per financial section (e.g., "Sales," "EBITDA").

### 4. Prediction
- For rows where `DEC 2023` is missing, the trained model predicts the value using the available historical data.

### 5. Evaluation
- The model is evaluated on rows where `DEC 2023` is available using:
  - **R² Score:** Measures how well the model explains the data.
  - **Median Absolute Error (MedAE):** Measures prediction accuracy.

### 6. Output
The final dataset includes both the original and predicted `DEC 2023` values.

---

## Implementation



### Installation
1. Clone the repository:
   ```bash
   git clone ...
