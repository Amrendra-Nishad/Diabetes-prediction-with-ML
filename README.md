# Diabetes Prediction with Machine Learning

A machine learning model that predicts whether a patient is likely to have diabetes based on diagnostic health measurements, using a Support Vector Machine (SVM) classifier.

## Overview

This project applies supervised machine learning to the classic **PIMA Indians Diabetes Dataset** to classify patients as diabetic or non-diabetic based on medical diagnostic features such as glucose level, blood pressure, BMI, insulin level, and age.

## Dataset

- **File:** `diabetes.csv`
- **Source:** PIMA Indians Diabetes Dataset
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Target:** Outcome (1 = Diabetic, 0 = Non-Diabetic)

## Approach

1. **Data Preprocessing** — Loaded and cleaned the dataset, handled feature scaling using `StandardScaler` to normalize the input features
2. **Train-Test Split** — Split the data into training and testing sets to evaluate model performance on unseen data
3. **Model Training** — Trained a **Support Vector Machine (SVM)** classifier with a linear kernel to separate diabetic and non-diabetic cases
4. **Evaluation** — Assessed model performance using accuracy score on both training and test sets

## Tech Stack

- **Python**
- **scikit-learn** — model building and evaluation
- **pandas / NumPy** — data handling
- **Jupyter Notebook** — development environment

## How to Run

```bash
# Clone the repository
git clone https://github.com/Amrendra-Nishad/Diabetes-prediction-with-ML.git
cd Diabetes-prediction-with-ML

# Install dependencies
pip install pandas numpy scikit-learn jupyter

# Launch the notebook
jupyter notebook "diabetes prediction.ipynb"
```

## Results

The trained SVM model achieves strong classification accuracy on the test set, correctly distinguishing between diabetic and non-diabetic patients based on their diagnostic measurements.

*(Add your exact accuracy score here once you check the notebook output, e.g. "Achieved 78% accuracy on the test set.")*

## Future Improvements

- Experiment with other algorithms (Random Forest, Logistic Regression) for comparison
- Add hyperparameter tuning (GridSearchCV) to optimize model performance
- Deploy the model as a simple web app for interactive predictions

## Author

**Amrendra Nishad**
[GitHub](https://github.com/Amrendra-Nishad)
