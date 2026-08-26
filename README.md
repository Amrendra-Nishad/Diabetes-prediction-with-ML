# Diabetes Prediction with Machine Learning

A machine learning model that predicts whether a patient is likely to have diabetes based on diagnostic health measurements, using Logistic Regression on a 100,000-row dataset.

## Overview

This project applies supervised machine learning to a diabetes diagnostic dataset, classifying patients as diabetic or non-diabetic based on age, BMI, HbA1c level, blood glucose level, hypertension, heart disease, gender, and smoking history.

## Dataset

File: `diabetes_prediction_dataset(1).csv`

The dataset has 100,000 rows, reduced to 96,146 after removing 3,854 duplicate rows. Features are age, gender, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, and blood_glucose_level, with `diabetes` (1 = diabetic, 0 = non-diabetic) as the target.

Only 8.5% of rows are labeled diabetic. That imbalance shaped most of the decisions below.

## Approach

The categorical columns (`gender`, `smoking_history`) were one-hot encoded with `pd.get_dummies`. The data was split into train and test sets before scaling, and `StandardScaler` was fit only on the training set, to avoid leaking test-set information into the scaler.

Because diabetic cases make up less than a tenth of the data, the model was trained with `class_weight='balanced'` — without it, a model can score well by mostly predicting "not diabetic" and still miss the minority class almost entirely. Logistic Regression was chosen over other classifiers because it directly outputs a calibrated probability for each prediction, which lets recall and precision get evaluated at a meaningful threshold rather than an arbitrary one, and because it stays interpretable: each feature has a coefficient that can be explained directly, rather than needing feature-importance approximations.

The model was evaluated on precision, recall, F1-score, and ROC-AUC rather than accuracy alone, since accuracy on its own is misleading on a dataset this imbalanced.

## Tech Stack

Python, scikit-learn for model building and evaluation, pandas and NumPy for data handling, and Jupyter Notebook for development.

## How to Run
Install dependencies

pip install pandas numpy scikit-learn jupyter

## Results

| Metric | Score |
|---|---|
| Accuracy | 89% |
| Diabetic Recall | 0.88 |
| Diabetic Precision | 0.43 |
| ROC-AUC | 0.961 |

The model correctly identifies 1,496 of 1,696 actual diabetic cases in the test set, missing only 200. That comes at the cost of a higher false-positive rate (1,989 people flagged who aren't actually diabetic) and a lower overall accuracy than a model tuned purely for that number would show. For a screening tool, that trade-off is intentional: missing a real diabetic case is a worse outcome than a false alarm that gets ruled out by a follow-up test.

## Future Improvements

Hyperparameter tuning with GridSearchCV, threshold tuning on the predicted probabilities instead of the default 0.5 cutoff to explicitly control the recall/precision balance, and deploying the model as a simple web app for interactive predictions.

## Author

Amrendra Nishad — [GitHub](https://github.com/Amrendra-Nishad)
