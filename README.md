# Machine Learning Capstone Project

## 23CSE301 – Machine Learning | B.Tech CSE | Academic Year 2026–27

This repository contains our Machine Learning Capstone Project for **23CSE301 – Machine Learning**. The project follows an end-to-end Machine Learning workflow including data loading, exploratory data analysis (EDA), data cleaning, feature engineering, preprocessing, model training, evaluation, and model comparison.

The capstone project consists of three machine learning tracks:

- **Regression**
- **Classification**
- **Clustering**

### Project Status

| Track | Status |
|---|---|
| Regression | ✅ Completed – Review 1 |
| Classification – Part A | ✅ Completed – Review 1 |
| Classification – Part B | 🔄 To be completed – Review 2 |
| Clustering | 🔄 To be completed – Review 2 |

---

# 1. Regression Track – Appliances Energy Prediction

## Problem Statement

The objective of this track is to predict household appliance energy consumption using environmental, weather, and time-related attributes from the dataset.

## Dataset

**Dataset:** Appliances Energy Prediction

- **Number of records:** 19,735
- **Number of features:** 29
- **Target variable:** `Appliances`
- **Target type:** Continuous numerical variable

The dataset contains information related to indoor environmental conditions, outdoor weather conditions, lighting, and time-related attributes that can be used to predict appliance energy consumption.

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded and inspected the dataset.
2. Checked dataset shape, data types, missing values, and duplicates.
3. Removed unnecessary attributes such as `rv1` and `rv2`.
4. Converted the `date` attribute into useful time-based features:
   - `hour`
   - `day`
   - `month`
   - `weekday`
5. Performed exploratory data analysis.
6. Detected potential outliers using the IQR method.
7. Applied IQR-based outlier capping.
8. Separated independent variables and target variable.
9. Performed an 80:20 train-test split.
10. Applied feature scaling where required by the algorithms.

## Regression Algorithms

The following ten regression algorithms were implemented:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regressor (SVR)
10. K-Nearest Neighbors Regressor

## Regression Evaluation

The models were evaluated using:

- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

All models were evaluated using the same held-out test set for fair comparison.

## Regression Results

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Random Forest Regression | 0.5988 | 61.7382 | 28.7907 |
| K-Nearest Neighbors Regression | 0.4897 | 69.6318 | 32.3142 |
| Gradient Boosting Regression | 0.3087 | 81.0436 | 44.0114 |
| Polynomial Regression | 0.3009 | 81.4999 | 48.6393 |
| Decision Tree Regression | 0.2149 | 86.3679 | 35.7411 |
| Ridge Regression | 0.1388 | 90.4553 | 52.3095 |
| Linear Regression | 0.1388 | 90.4569 | 52.3143 |
| Lasso Regression | 0.1284 | 90.9990 | 51.9923 |
| ElasticNet Regression | 0.0852 | 93.2273 | 53.6184 |
| Support Vector Regression | 0.0241 | 96.2886 | 41.4862 |

## Regression Visualizations

The regression notebook contains visualizations including:

- Target distribution
- Feature distributions
- Correlation heatmap
- Feature-target scatter plots
- Outlier visualizations
- Model performance comparison

---

# 2. Classification Track – Telco Customer Churn

## Problem Statement

The classification task aims to predict whether a telecom customer is likely to **churn** based on customer demographics, tenure, subscribed services, contract information, billing details, and other customer attributes.

## Dataset

**Dataset:** Telco Customer Churn

- **Number of records:** 7,043
- **Target variable:** `Churn`
- **Classes:** `Yes` and `No`
- **Problem type:** Binary Classification

## Data Preprocessing

The following preprocessing and feature engineering steps were performed:

1. Loaded and inspected the dataset.
2. Converted `TotalCharges` into a numeric data type.
3. Handled invalid/missing values in `TotalCharges`.
4. Removed the `customerID` identifier.
5. Created the `avg_monthly_spend` feature.
6. Created the `num_services_subscribed` feature.
7. Created the `is_month_to_month` feature.
8. Encoded the target variable:
   - `No → 0`
   - `Yes → 1`
9. Identified numerical and categorical attributes.
10. Performed an 80:20 stratified train-test split.
11. Applied standard scaling to numerical features.
12. Applied one-hot encoding to categorical features.

## Classification – Part A Algorithms

The following five algorithms were implemented for Review 1:

1. Logistic Regression
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Classifier (SVC)

## Classification Evaluation

The Part A models were evaluated using:

- Accuracy
- Weighted F1-score
- Confusion Matrix

Five-fold cross-validation was also performed for the selected models.

## Classification Results

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Logistic Regression | 0.8062 | 0.8013 |
| Decision Tree Classifier | 0.7991 | 0.7965 |
| SVC | 0.7963 | 0.7870 |
| K-Nearest Neighbors | 0.7672 | 0.7666 |
| Naive Bayes | 0.6842 | 0.7025 |

## Cross-Validation Results

| Model | Mean Weighted F1 | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.7992 | 0.0132 |
| Decision Tree Classifier | 0.7802 | 0.0196 |

## Classification Visualizations

The classification notebook includes:

- Confusion matrices
- Model comparison
- Feature analysis
- Classification performance evaluation

---

# 3. Overall Machine Learning Workflow

The project follows the following workflow:

```text
Dataset
   ↓
Data Loading
   ↓
Data Audit
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Encoding & Scaling
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Model Comparison
