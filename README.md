# Bankruptcy Prediction Model

## Overview
This project implements a bankruptcy prediction model using machine learning techniques. The dataset consists of financial indicators of companies, and the goal is to predict whether a company will go bankrupt based on these features. 

The model uses multiple classifiers, including:
- **Random Forest**
- **XGBoost**
- **LightGBM**

## Features
The dataset includes various financial ratios such as:
- **Liquidity Ratios** (Current Ratio, Quick Ratio)
- **Profitability Ratios** (Return on Assets, Net Profit Margin)
- **Solvency Ratios** (Debt-to-Equity Ratio)
- **Efficiency Ratios** (Accounts Receivable Turnover) and more...

## Data Preprocessing
The data is preprocessed using:
- Handling missing values (if any)
- Handling class imbalance using **SMOTE**

## Model Training and Evaluation
Each classifier is trained on the preprocessed data, and their performance is evaluated based on:
- **Accuracy**
- **Classification Report**
- **Overfitting Check**

## Running the Model
1. Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn
    ```
2. Load and preprocess the dataset.
3. Train the models using the provided script.
4. Evaluate predictions on test data.

## Predictions
The trained models generate predictions indicating whether a company is likely to go bankrupt (`1`) or not (`0`).

## License
This project is licensed under the **MIT License**. You are free to use, modify, and distribute this project as long as proper credit is given.
