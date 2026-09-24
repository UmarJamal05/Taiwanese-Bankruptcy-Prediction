# Corporate Bankruptcy Prediction

## Overview
This project predicts the probability that a company will go bankrupt, using year-end
financial ratios. The dataset covers 6,819 companies from the Taiwan Economic Journal
(1999-2009), described by 94 financial ratios, with 220 bankruptcies (3.23% of the
sample) — a severely imbalanced classification problem.

The final model is **LightGBM with class weighting, calibrated with Platt scaling**,
using a decision threshold chosen explicitly rather than the default 0.5.

## Features
The dataset includes financial ratios across several categories, including:
- **Profitability** (ROA, net income to total assets, operating margins)
- **Leverage / solvency** (debt ratio, net worth to assets)
- **Liquidity** (current ratio, quick ratio, working capital)
- **Efficiency and turnover** (asset turnover, receivables turnover)

## Methodology
- **Exploratory analysis** — class balance, correlation with the target, and
  multicollinearity among the ratios.
- **Cleaning** — dropped one constant column; confirmed no missing values or duplicates.
- **Stratified 80/20 train/test split**, with the test set held out until final evaluation.
- **Baselines** — dummy classifier and logistic regression, as a floor for comparison.
- **Imbalance handling** — class weights and several resampling strategies (SMOTE,
  BorderlineSMOTE, ADASYN, SMOTETomek, random undersampling) compared fairly, inside
  cross-validation folds, to avoid leakage.
- **Model comparison** — Logistic Regression, Random Forest, XGBoost, and LightGBM,
  under repeated stratified cross-validation, selected by **average precision (PR-AUC)**
  rather than accuracy (accuracy is a poor metric at 3% positive prevalence).
- **Hyperparameter tuning** — `RandomizedSearchCV`, re-validated under the same CV
  scheme used for model selection.
- **Probability calibration** — Platt scaling (`CalibratedClassifierCV`, sigmoid).
- **Decision threshold selection** — chosen on out-of-fold predictions via F-beta
  optimisation and an explicit expected-cost calculation, not left at 0.5.
- **Interpretability** — feature importance and SHAP values for individual predictions.
- **Persistence** — the fitted model, threshold, and feature metadata are saved together
  as one bundle, with a ready-to-use inference function.

## Results

| Metric | Value |
|---|---|
| Dataset | 6,819 companies, 94 features |
| Class balance | 3.23% bankrupt (220 cases) |
| Final model | LightGBM (class-weighted) + Platt calibration |
| Selection metric | Average precision (PR-AUC), 5-fold stratified CV |
| Decision threshold | 0.0240 (median F2-optimal, out-of-fold) |
| Test PR-AUC | 0.565 (17.5× random) |
| Test ROC-AUC | 0.962 |
| Test Brier score | 0.022 |
| Test recall | 0.705 (31 of 44 bankruptcies caught) |
| Test precision | 0.456 (37 false alarms among 1,320 healthy companies) |

Note: the model's ranking quality (PR-AUC, ROC-AUC) is stable across runs. The exact
decision threshold — and therefore the recall/precision split above — can shift slightly
run to run, since it's selected from a precision-recall curve that's inherently noisy
with only ~176 positive training examples.

## Running the project
1. Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn lightgbm xgboost imbalanced-learn shap matplotlib seaborn joblib
    ```
2. Place `data.csv` in the same directory as the notebook.
3. Run `bankruptcy_prediction.ipynb` top to bottom.
4. The final cell saves a model bundle (model + threshold + feature metadata) that the
   included `predict_bankruptcy_risk()` function uses to score new companies.

## Predictions
The trained model outputs a bankruptcy probability for each company; the bundled
threshold converts that into a `0`/`1` flag calibrated for a cost-sensitive use case
(missing a bankruptcy is treated as costlier than a false alarm).

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
