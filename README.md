# Financial Forecasting

## Overview
First part of this project uses **KNN, Lasso, Ridge, Linear Regression** to predict financial metrics (e.g., `DEC 2023`) based on historical data (`DEC 2019–2022`). It handles cases where some rows already have `DEC 2023` values by:
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


# MAML-Based Financial Indicator Prediction Model

## Overview
This repository contains an implementation of **Model-Agnostic Meta-Learning (MAML)** for predicting **buy-side financial indicators** based on **sell-side financial indicators** across multiple deals. The model is designed to handle **multi-task learning**, where each financial deal is treated as a separate task.

## Features
- **Multi-task Learning Approach**: Each financial deal is considered a separate learning task.
- **Memory-Augmented Neural Network (MANN)**: Utilizes **enhanced memory retrieval and attention-based updates**.
- **Standardization & Scaling**: Sell-side and buy-side financial indicators are standardized for each task.
- **Custom Training Strategy**: Implements curriculum learning and early stopping.
- **Residual Analysis**: Plots **actual vs. predicted errors** for financial indicators.

## Dataset
The dataset consists of financial indicators over five years:

| SECTION | SELL_DEC 2019 | SELL_DEC 2020 | SELL_DEC 2021 | SELL_DEC 2022 | SELL_DEC 2023 | BUY_DEC 2019 | BUY_DEC 2020 | BUY_DEC 2021 | BUY_DEC 2022 | BUY_DEC 2023 |
|---------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| 1 | Cash & Short-Term Investments | 94.08087 | 93.49270 | 12.99180 | 1.86473 | 1.00914 | 165.43600 | 215.95100 | 298.45400 | 385.6700 | 277.91100 |
| 1 | EBIT | -3.99321 | 0.96715 | 3.03496 | -1.37326 | 10.05510 | 196.27000 | 158.80000 | 270.41300 | 302.0590 | 303.02500 |
| 1 | EBITDA | -2.62899 | 3.95140 | 6.77084 | 3.87702 | 12.24520 | 271.46900 | 237.03200 | 350.25200 | 390.2510 | 424.72800 |
| 1 | Gross Income | 45.19710 | 41.66335 | 15.94888 | 36.99234 | 19.63104 | 349.23800 | 277.98600 | 414.24700 | 467.8340 | 478.41800 |
| 1 | Net Debt | -94.08087 | -92.18860 | -12.99180 | -0.98175 | -1.00914 | 746.78600 | 488.06400 | 314.79800 | 198.5270 | 241.39200 |

## Dependencies
Make sure you have the following libraries installed:
```bash
pip install pandas numpy torch tensorflow scikit-learn xgboost statsmodels matplotlib
```

## Data Preprocessing
### Steps:
1. **Load Dataset:** Read the financial indicators dataset.
2. **Replace Missing Values:** Convert `"-"` to `NaN` and drop missing values.
3. **Convert Data Types:** Ensure numerical columns are properly formatted.
4. **Standardization:** Normalize sell-side and buy-side indicators for each deal.
5. **Task Creation:** Each financial deal is treated as an independent learning task.

## Task Preparation
Each financial deal is treated as a separate task. Standardization is applied to both **sell-side** and **buy-side** indicators using `StandardScaler`.

```python
# Prepare structured tasks for MAML
from sklearn.preprocessing import StandardScaler

# Create tasks
tasks = []
task_scalers = {}

for deal_id in df_filtered['N'].unique():
    deal_data = df_filtered[df_filtered['N'] == deal_id]
    X_sell = deal_data[sell_side_cols].values
    Y_buy = deal_data[buy_side_cols].values
    scaler_sell, scaler_buy = StandardScaler(), StandardScaler()
    X_scaled, Y_scaled = scaler_sell.fit_transform(X_sell), scaler_buy.fit_transform(Y_buy)
    tasks.append((X_scaled, Y_scaled))
    task_scalers[deal_id] = scaler_buy
```

## Model Architecture
The **Memory-Augmented Neural Network (MANN)** with **feedforward layers** is implemented using TensorFlow.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class MANN_fnn(Model):
    def __init__(self, input_dim, output_dim, memory_size=400, embedding_dim=128):
        super(MANN_fnn, self).__init__()
        self.memory_keys = tf.Variable(tf.random.normal([memory_size, input_dim]), trainable=False)
        self.memory_values = tf.Variable(tf.random.normal([memory_size, output_dim]), trainable=False)
        self.controller = tf.keras.Sequential([
            layers.Dense(128, activation='swish'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='swish'),
            layers.LayerNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='swish'),
            layers.LayerNormalization()
        ])
        self.fc = layers.Dense(output_dim)
```

## Training the Model
The model is trained using an **advanced strategy** with a **cosine decay learning rate schedule**, **temporal regularization**, and **early stopping**.

```python
def train_mann_fnn(model, train_tasks, val_tasks, epochs=100):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5)
```

## Model Evaluation
### R² Score Calculation
The model is evaluated using **R² score** for multiple validation tasks.

```python
from sklearn.metrics import r2_score
r2_scores = []
for i, task_id in enumerate(val_sample_ids):
    X_test, Y_true = val_tasks[i]
    Y_pred = maml_regressor.predict(X_test)
    r2 = r2_score(Y_true, Y_pred)
    r2_scores.append(r2)
print(f"Average Validation R² Score: {np.mean(r2_scores):.4f}")
```

### Residual Analysis
Residual analysis helps assess model performance by plotting actual vs. predicted errors.

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Sample ID")
plt.ylabel("Residual (Actual - Predicted)")
plt.show()
```

## Conclusion
This project demonstrates the power of **meta-learning** for financial indicator predictions, utilizing **task-based training**, **memory-enhanced neural networks**, and **advanced training strategies** to optimize model performance.

## Further Improvements
Here are some areas where the model can be improved:
- **Feature Engineering**: Investigate additional financial metrics that could improve model performance.
- **Advanced Neural Architectures**: Explore Transformer-based architectures or attention mechanisms for better temporal learning.
- **Hyperparameter Optimization**: Use Bayesian Optimization or Grid Search to fine-tune model hyperparameters.
- **Larger Dataset Integration**: Incorporate more financial deals to enhance the generalization capability.
- **Explainability & Interpretability**: Implement SHAP or LIME to understand how features impact predictions.
- **Deployment & API Integration**: Develop an API for real-time predictions and integrate with financial analytics dashboards.

## Contributions
Feel free to contribute to this project by submitting pull requests or opening issues for discussion!

## License
This project is licensed under the MIT License - see the LICENSE file for details.
