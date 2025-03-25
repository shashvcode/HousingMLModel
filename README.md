# California Housing Price Prediction

This project uses the California Housing dataset to predict median house values based on various housing and demographic features.

## Overview

A complete machine learning pipeline was built using Scikit-learn and XGBoost, including:

- **Data Preprocessing**:  
  - Handled missing values with median imputation  
  - Scaled features using `StandardScaler`  
  - Engineered new features (e.g., rooms per household)

- **Feature Selection**:  
  - Created meaningful ratios and dropped noisy features to reduce overfitting

- **Modeling Techniques**:  
  - Trained multiple models:  
    - `LinearSVR`  
    - `Ridge Regression`  
    - `Random Forest Regressor`  
    - `XGBoost Regressor`  
    - `SVR (RBF)`  
  - Tuned models with `GridSearchCV` and `RandomizedSearchCV`
  - Built a final **Stacking Regressor** combining multiple base models

## Current Best Model

- **Best model**: Tuned `XGBoostRegressor`
- **Performance**: RMSE ≈ **18,744** on training set, **≈ 43,000** on test, likely due to overifitting

## Continuation
- ** I am continuin to experiemnt with new features, selecting more optimized models, and tuning hyparparameters to improve performance**
