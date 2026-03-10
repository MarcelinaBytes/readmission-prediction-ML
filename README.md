# readmission-prediction-ML
Machine learning project that predicts 30‑day hospital readmission risk using the UCI Diabetes readmission dataset. Includes EDA, modeling, and SHAP model explainability.

# Hospital Readmission Prediction (30-Day Readmission Risk)

This project builds a machine learning model to predict 30-day hospital readmission using the UCI Diabetes 130-Hospital Readmission dataset.

# Project Overview 
Hospital readmissions are a major quality and cost challenge in U.S. healthcare. Accurately identifying patients at high risk of being readmitted within 30 days allows clinicians, care teams, and hospital administrators to:

- Improve patient outcomes
- Allocate resources more effectively
- Reduce preventable readmissions
- Support value‑based care initiatives

This project uses machine learning to predict whether a patient will be readmitted within 30 days based on demographics, diagnoses, lab values, medications, comorbidities, and prior hospital utilization.
The goal is to build an interpretable and clinically meaningful model suitable for real‑world hospital settings.

## Dataset

**Source:** UCI Machine Learning Repository  
**Dataset:** “Diabetes 130-US Hospitals for Years 1999–2008”

**Rows:** 101,766 encounters  

**Features include:**
- Diagnoses (ICD-9 codes)
- Lab values & abnormal flags
- Procedures
- Medications
- Demographics
- Encounter type
- Prior admissions
- Length of stay
- Discharge disposition

## Target Variable

The original dataset includes a column called **`readmitted`**, which captures whether a patient was readmitted after discharge. It contains three possible values:

- **`NO`** – the patient was not readmitted  
- **`>30`** – the patient was readmitted more than 30 days after discharge  
- **`<30`** – the patient was readmitted within 30 days  

For this project, the goal is to predict *clinically important* readmissions — those occurring **within 30 days**, which are often used as a quality-of-care metric.

###Transformation for Modeling

To convert the original 3‑category variable into a binary classification target:

- **`<30` → 1** (positive class: 30‑day readmission)  
- **`NO` → 0**  
- **`>30` → 0**  

This allows us to build a model that predicts **30‑day readmission risk**, which aligns with hospital performance reporting and CMS readmission measures.

# Project Objectives 
- Clean and preprocess clinical data
- Perform exploratory data analysis (EDA)
- Train ML models to classify readmission risk
- Evaluate performance using ROC‑AUC, F1, precision/recall
- Interpret model using SHAP values
- Package model and create a simple Streamlit app

# Project Structure
```
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
```

# Future Enhancements
- Incorporate temporal modeling (RNN/LSTM or transformer‑based)
- Deploy full API with FastAPI
- Add dashboards for real‑time clinical monitoring
- Validate model on MIMIC‑IV
