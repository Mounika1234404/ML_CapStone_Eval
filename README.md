Machine Learning Capstone Project

23CSE301 – Machine Learning | B.Tech CSE | Academic Year 2026–27

This repository contains the Machine Learning Capstone Project for 23CSE301 – Machine Learning. The project follows an end-to-end machine learning workflow covering data loading, exploratory data analysis (EDA), data cleaning, feature engineering, preprocessing, model training, evaluation, and comparison.

The capstone is organized around three machine learning tracks:

Regression

Classification

Clustering

Current Project Status

Track

Current Status

Regression

✅ Completed for Review 1

Classification – Part A

✅ Completed for Review 1

Classification – Part B

🔄 Planned for Review 2

Clustering

🔄 Planned for Review 2

1. Regression Track – Appliances Energy Prediction

Problem Statement

Household appliance energy consumption varies with environmental conditions, weather, and time of day. The objective of this track is to predict appliance energy consumption using environmental, weather, and time-related features.

Dataset

Dataset: Appliances Energy Prediction

Rows: 19,735

Original features: 29

Target variable: Appliances

Target type: Continuous numerical value

Data quality: No missing values and no duplicate records were found in the provided dataset.

Preprocessing and Feature Engineering

The regression pipeline includes:

Dataset loading and inspection

Missing-value check

Duplicate-record check

Removal of rv1 and rv2

Conversion of date into useful time features:

hour

day

month

weekday

Separation of features and target

Exploratory data analysis

Outlier detection using the IQR method

IQR-based outlier capping using training-set statistics

80:20 train-test split with stratification based on target quantile bins

Standard scaling for models requiring scaled features

Regression Algorithms

The following 10 required algorithms were implemented:

Linear Regression

Ridge Regression

Lasso Regression

ElasticNet Regression

Polynomial Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Support Vector Regressor (SVR)

K-Nearest Neighbors Regressor

Regression Results

All models were evaluated on the same held-out test set using R², RMSE, and MAE.

Model

R²

RMSE

MAE

Random Forest Regression

0.5988

61.7382

28.7907

K-Nearest Neighbors Regression

0.4897

69.6318

32.3142

Gradient Boosting Regression

0.3087

81.0436

44.0114

Polynomial Regression

0.3009

81.4999

48.6393

Decision Tree Regression

0.2149

86.3679

35.7411

Ridge Regression

0.1388

90.4553

52.3095

Linear Regression

0.1388

90.4569

52.3143

Lasso Regression

0.1284

90.9990

51.9923

ElasticNet Regression

0.0852

93.2273

53.6184

Support Vector Regression

0.0241

96.2886

41.4862

Regression Visualizations

The notebook includes:

Target distribution

Feature distribution plots

Correlation heatmap

Lights vs. appliance consumption scatter plot

Outdoor temperature vs. appliance consumption scatter plot

Box plots for outlier detection

R² comparison across the ten regression models

2. Classification Track – Telco Customer Churn

Problem Statement

The classification task focuses on predicting whether a telecom customer is likely to churn based on demographic information, tenure, subscribed services, contract details, billing information, and other customer attributes.

Dataset

Dataset: Telco Customer Churn

Rows: 7,043

Target variable: Churn

Classes: No, Yes

Classification type: Binary classification

Preprocessing and Feature Engineering

The classification pipeline includes:

Conversion of TotalCharges from string/object representation to numeric values

Handling of invalid/missing TotalCharges values

Removal of the identifier column customerID

Creation of avg_monthly_spend

Creation of num_services_subscribed

Creation of is_month_to_month

Encoding of the target variable:

No → 0

Yes → 1

Identification of categorical and numerical features

80:20 stratified train-test split

Standard scaling of numerical features

One-hot encoding of categorical features

Classification – Part A Algorithms

The following five algorithms required for Review 1 were implemented:

Logistic Regression

K-Nearest Neighbors

Gaussian Naive Bayes

Decision Tree Classifier

Support Vector Classifier (SVC)

Classification Results

The Part A models were evaluated using Accuracy and Weighted F1-score.

Model

Accuracy

Weighted F1

Logistic Regression

0.8062

0.8013

Decision Tree Classifier

0.7991

0.7965

SVC

0.7963

0.7870

K-Nearest Neighbors

0.7672

0.7666

Naive Bayes

0.6842

0.7025

Cross-Validation

Five-fold cross-validation was performed on the two leading Part A models:

Model

Mean Weighted F1

Standard Deviation

Logistic Regression

0.7992

0.0132

Decision Tree Classifier

0.7802

0.0196

Classification Visualizations

The notebook includes confusion matrices for all five Part A classifiers.

3. Machine Learning Workflow

The overall workflow followed in the project is:

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

4. Repository Structure

The repository follows the capstone project structure:

ML-Capstone-Project/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── energydata.csv
│   └── telco_churn.csv
│
├── notebooks/
│   ├── Appliances_Energy_Prediction.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
│
├── models/
│   └── # saved models, if applicable
│
└── app/
    └── # GUI/deployment files, if applicable

clustering.ipynb, saved models, and application/deployment files will be added as the corresponding parts of the capstone are completed.

5. Technologies Used

Python 3

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Jupyter Notebook / Google Colab

6. Installation

Clone the repository and install the required Python packages:

git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
pip install -r requirements.txt

If requirements.txt is not available yet, the main dependencies can be installed using:

pip install pandas numpy scikit-learn matplotlib seaborn jupyter

7. How to Run

Regression

Open:

notebooks/Appliances_Energy_Prediction.ipynb

Make sure the energy dataset is available at the path used by the notebook.

Run all cells from top to bottom.

The notebook performs EDA, preprocessing, trains all ten regression models, and displays the comparison results.

Classification

Open:

notebooks/classification.ipynb

Make sure the Telco Churn dataset is available at the path expected by the notebook.

Run all cells from top to bottom.

The notebook performs preprocessing, feature engineering, trains the five Part A classifiers, generates confusion matrices, compares the models, and performs five-fold cross-validation on the selected models.

Note: The current classification notebook uses a Google Colab/Google Drive path. If running directly from GitHub/Jupyter, update the dataset path to the location of the dataset in the repository.

8. Reproducibility

A fixed random state of 42 is used wherever applicable to make the experiments reproducible.

The same train-test split is used for model comparison within each track so that the reported metrics can be compared consistently.

Preprocessing transformations such as scaling and categorical encoding are fitted on the training data and then applied to the test data.

9. Future Work – Review 2

The remaining capstone components will extend the current work with:

Classification – Part B

Random Forest Classifier

AdaBoost Classifier

Gradient Boosting Classifier

Bagging Classifier

MLP Classifier

The final classification comparison will include all ten required algorithms and the required metrics:

Accuracy

Precision

Recall

Weighted F1-score

ROC-AUC

Confusion matrix

Clustering

The clustering track will include:

K-Means Clustering

Agglomerative Hierarchical Clustering

Elbow curve

Dendrogram

Silhouette Score

Davies-Bouldin Index

Calinski-Harabasz Index

PCA-based cluster visualization

t-SNE visualization

10. Project Guidelines

This project is developed according to the 23CSE301 Machine Learning Capstone Project Guidelines, Algorithm List & Evaluation Rubrics.

The guidelines require an end-to-end ML pipeline, consistent evaluation, required visualizations, model comparison, reproducibility, and a structured GitHub repository.

11. AI Assistance

Generative AI tools were used for limited code-scaffolding and documentation support. The final notebooks, preprocessing decisions, model implementation, results, and project interpretation should be reviewed and finalized by the project team in accordance with the course's academic-integrity guidelines.

12. Team

Course: 23CSE301 – Machine Learning
Program: B.Tech. Computer Science and Engineering
Academic Year: 2026–27
Team Size: 3 members

License

This project is developed as part of an academic Machine Learning Capstone Project.
