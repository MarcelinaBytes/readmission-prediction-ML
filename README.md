# readmission-prediction-ML
Machine learning project that predicts 30‑day hospital readmission risk using the UCI Diabetes readmission dataset. Includes EDA, modeling, and SHAP model explainability.

# Hospital Readmission Prediction (30-Day Readmission Risk)

This project builds a machine learning model to predict 30-day hospital readmission using the UCI Diabetes 130-Hospital Readmission dataset.

# Project Overview 
Hospital readmissions are a major quality and cost challenge in U.S. healthcare. Accurately identifying patients at high risk of being readmitted within 30 days allows clinicians, care teams, and hospital administrators to:

Improve patient outcomes
Allocate resources more effectively
Reduce preventable readmissions
Support value‑based care initiatives

This project uses machine learning to predict whether a patient will be readmitted within 30 days based on demographics, diagnoses, lab values, medications, comorbidities, and prior hospital utilization.
The goal is to build an interpretable and clinically meaningful model suitable for real‑world hospital settings.

# Dataset
Source: UCI Machine Learning Repository
Dataset: “Diabetes 130‑US Hospitals for Years 1999–2008”
Rows: 101,766 encounters
Features:
Diagnoses (ICD‑9 codes)
Lab values and abnormal flags
Procedures
Medications
Demographics
Encounter type
Prior admissions
Length of stay
Discharge disposition

Target variable: readmitted (transformed to: readmitted within 30 days = 1, otherwise = 0)

# Project Objectives 
Clean and preprocess clinical data
Perform exploratory data analysis (EDA)
Train ML models to classify readmission risk
Evaluate performance using ROC‑AUC, F1, precision/recall
Interpret model using SHAP values
Package model and create a simple Streamlit app

# Project Structure
readmission-prediction-ML/
│── README.md
│── requirements.txt
│── data/
│   ├── raw/
│   └── processed/
│── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
│── src/
│   ├── data_prep.py
│   ├── train_model.py
│   └── evaluate.py
│── models/
│── results/
│   ├── charts/
│   └── shap/
└── app/
    └── app.py

# Future Enhancements
Incorporate temporal modeling (RNN/LSTM or transformer‑based)
Deploy full API with FastAPI
Add dashboards for real‑time clinical monitoring
Validate model on MIMIC‑IV
