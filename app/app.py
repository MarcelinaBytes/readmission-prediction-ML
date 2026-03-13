# app/app.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- App Config ----------
st.set_page_config(page_title="30-Day Readmission Risk", page_icon="🏥", layout="centered")

st.title("🏥 30‑Day Readmission Risk Predictor")
st.caption("Demo app for portfolio — uses the trained pipeline from Phase 5")

# ---------- Load artifacts ----------
@st.cache_resource
def load_pipeline():
    path = os.path.join("models", "best_readmission_model.pkl")
    pipe = joblib.load(path)  # sklearn Pipeline(preprocess, clf)
    return pipe

@st.cache_data
def load_reference_data():
    # Used only to pre-populate widget choices (not for model inference)
    path = os.path.join("data", "processed", "clean_step1.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

pipe = load_pipeline()
preprocess = pipe.named_steps["preprocess"]
clf = pipe.named_steps["clf"]

df_ref = load_reference_data()

# The exact feature names the ColumnTransformer expects:
model_feature_cols = list(preprocess.feature_names_in_)

# Split what we *intend* to collect from user
# (match what you used in Phase 5; we'll intersect with model_feature_cols)
intended_numeric = [
    "time_in_hospital","num_lab_procedures","num_procedures","num_medications",
    "number_outpatient","number_emergency","number_inpatient","prior_utilization","a1c_score"
]
# include dx counts if present in model
intended_numeric += [c for c in model_feature_cols if c.startswith("diag_count_")]

intended_categorical = [
    "gender","age","admission_type_id","discharge_disposition_id","admission_source_id","insulin"
]

# Keep only those the pipeline truly expects
numeric_cols = [c for c in intended_numeric if c in model_feature_cols]
categorical_cols = [c for c in intended_categorical if c in model_feature_cols]
ordered_cols = numeric_cols + categorical_cols

st.write("**Model expects features:**")
st.code(ordered_cols, language="python")

st.markdown("---")

# ---------- Inputs → UI ----------
st.subheader("Enter Patient Encounter Features")

def opt_list(col, default=None):
    """Build option list from reference data if available; else simple defaults."""
    if df_ref is not None and col in df_ref.columns:
        opts = sorted([str(x) for x in pd.Series(df_ref[col].dropna().unique()).astype(str)])
        return opts
    # Fallback defaults (sane options)
    presets = {
        "gender": ["Male","Female","Unknown/Invalid","Unknown","Other"],
        "age": ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)",
                "[60-70)","[70-80)","[80-90)","[90-100)"],
        "insulin": ["No","Steady","Up","Down"]
    }
    if col in presets:
        return presets[col]
    return []

# Numeric widgets
num_inputs = {}
if len(numeric_cols) > 0:
    with st.container():
        st.write("### Numeric Features")
        for col in numeric_cols:
            # provide a reasonable slider range; you can tune if you like
            if col == "time_in_hospital":
                val = st.slider(col, 1, 14, 4)
            elif col in ("num_lab_procedures","num_procedures","num_medications"):
                val = st.slider(col, 0, 100 if col!="num_procedures" else 20, 10 if col!="num_procedures" else 1)
            elif col in ("number_outpatient","number_emergency","number_inpatient","prior_utilization"):
                val = st.slider(col, 0, 20, 0)
            elif col == "a1c_score":
                # 0=None, 1=Normal, 2=>7, 3=>8 from earlier mapping
                val = st.slider(col, 0, 3, 0)
            else:
                # generic numeric
                val = st.number_input(col, min_value=0.0, value=0.0, step=1.0)
            num_inputs[col] = val

# Categorical widgets
cat_inputs = {}
if len(categorical_cols) > 0:
    with st.container():
        st.write("### Categorical Features")
        for col in categorical_cols:
            choices = opt_list(col)
            if len(choices) > 0:
                # Use a sensible default value
                default_idx = 0
                if df_ref is not None and col in df_ref.columns:
                    # choose the most common category as default if possible
                    top = df_ref[col].value_counts().index[0]
                    if str(top) in choices:
                        default_idx = choices.index(str(top))
                val = st.selectbox(col, choices, index=default_idx)
            else:
                # fallback text input if we have no choices
                val = st.text_input(col, "")
            cat_inputs[col] = val

st.markdown("---")

# ---------- Predict ----------
if st.button("Predict readmission risk"):
    # Build a single-row DataFrame with the exact columns in the same order
    data = {}
    # Fill numeric
    for c in numeric_cols:
        data[c] = [num_inputs[c]]
    # Fill categorical
    for c in categorical_cols:
        data[c] = [cat_inputs[c]]

    X_user = pd.DataFrame(data)[ordered_cols]  # ensure exact order

    # Predict probability
    proba = float(pipe.predict_proba(X_user)[0, 1])
    pct = round(proba * 100, 1)

    # Simple risk bands (tune if desired)
    if proba < 0.15:
        band = "Low"
        color = "🟢"
    elif proba < 0.35:
        band = "Moderate"
        color = "🟡"
    else:
        band = "High"
        color = "🔴"

    st.success(f"**Predicted 30‑day readmission risk:** {pct}%  —  {color} **{band}**")

    with st.expander("See input row as the model saw it"):
        st.dataframe(X_user)

st.markdown("---")

# ---------- Optional: show pre-generated SHAP plots ----------
st.subheader("Explainability (Global)")
beeswarm = os.path.join("results","shap","summary_beeswarm.png")
barplot  = os.path.join("results","shap","summary_bar.png")

col1, col2 = st.columns(2)
with col1:
    if os.path.exists(beeswarm):
        st.image(beeswarm, caption="SHAP Beeswarm (Global Importance)", use_column_width=True)
    else:
        st.info("Beeswarm plot not found. Generate it in Phase 6 and commit to results/shap/.")

with col2:
    if os.path.exists(barplot):
        st.image(barplot, caption="SHAP Bar Plot (Global Importance)", use_column_width=True)
    else:
        st.info("Bar plot not found. Generate it in Phase 6 and commit to results/shap/.")
