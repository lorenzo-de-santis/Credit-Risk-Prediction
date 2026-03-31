# Credit Risk Prediction with Machine Learning

## Overview

This project develops a machine learning pipeline to predict the probability of borrower default (financial distress within 2 years).

## Key Contributions

* KNN-based imputation for missing values (~20% income missing)
* Custom feature scaling and transformation for outlier handling
* Feature engineering (e.g., non-real estate loans)
* Comparison of multiple non-linear models

## Results

* Models achieve similar performance: **AUC ≈ 0.86–0.87**
* Tree-based models outperform simpler approaches
* Key predictors:

  * credit utilization
  * past delinquencies
* Evidence of **non-random missing data** (linked to financial distress)

## Models

* Balanced Random Forest
* XGBoost
* Neural Network (with resampling for class imbalance)

## Structure

* `data/` → dataset and description
* `credit-risk-pipeline.ipynb` → full pipeline
* `credit-risk-modeling.ipynb` → modeling and evaluation

## Tech Stack

Python · Scikit-learn · XGBoost · Neural Networks
