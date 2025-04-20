import streamlit as st
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Dummy models for demo (you can replace with real joblib-loaded models)
sbp_model = GradientBoostingRegressor()
dbp_model = GradientBoostingRegressor()

# Train dummy models on synthetic data for testing
from sklearn.datasets import make_regression
X, y_sbp = make_regression(n_samples=100, n_features=5, noise=0.1)
_, y_dbp = make_regression(n_samples=100, n_features=5, noise=0.1)
sbp_model.fit(X, y_sbp)
dbp_model.fit(X, y_dbp)

def extract_hrv(ppg_waveform):
    ibi = np.diff(np.array(ppg_waveform))
    if len(ibi) < 2:
        return 0.0
    return np.sqrt(np.mean(np.square(np.diff(ibi))))

def predict_bp(hr, age, bmi, rr, ppg):
    hrv = extract_hrv(ppg)
    features = np.array([[hr, rr, age, bmi, hrv]])
    sbp = sbp_model.predict(features)[0]
    dbp = dbp_model.predict(features)[0]
    return round(sbp, 1), round(dbp, 1), 0.92  # Placeholder confidence

# Streamlit UI
st.set_page_config(page_title="BP Estimator", layout="centered")
st.title("\U0001F489 Blood Pressure Estimation App")
st.markdown("Enter the physiological values and PPG waveform below:")

hr = st.number_input("Heart Rate (BPM)", min_value=30, max_value=200, value=75)
age = st.number_input("Age", min_value=1, max_value=120, value=35)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
rr = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=40, value=16)

ppg_input = st.text_area("Enter PPG waveform as comma-separated values", "0.01, 0.03, 0.04, 0.06, 0.05, 0.04, 0.03")

if st.button("\U0001F52C Predict Blood Pressure"):
    try:
        ppg_waveform = [float(val.strip()) for val in ppg_input.split(",") if val.strip()]
        sbp, dbp, conf = predict_bp(hr, age, bmi, rr, ppg_waveform)
        st.success(f"\U0001FA78 Predicted Systolic BP: {sbp} mmHg")
        st.success(f"\U0001FA78 Predicted Diastolic BP: {dbp} mmHg")
        st.info(f"\U0001F4CA Confidence: {conf * 100:.1f}%")
    except Exception as e:
        st.error(f"\u274C Error processing input: {e}")
