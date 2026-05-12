import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1 {
    font-family: 'DM Serif Display', serif;
    color: #ff6b6b;
    text-align: center;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.stButton > button {
    background: linear-gradient(90deg, #ff6b6b, #ee0979);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    width: 100%;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(238,9,121,0.4);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(238,9,121,0.6);
}

.result-box {
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1.5rem;
    font-size: 1.4rem;
    font-weight: 600;
    animation: fadeIn 0.5s ease;
}

.result-positive {
    background: linear-gradient(135deg, #ff4e50, #f9d423);
    color: #1a1a2e;
}

.result-negative {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: #1a1a2e;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

label, .stSelectbox label, .stNumberInput label, .stSlider label {
    color: #1a1a1a !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

.section-header {
    color: #ff6b6b;
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    border-bottom: 1px solid rgba(255,107,107,0.3);
    padding-bottom: 0.3rem;
    margin: 1.2rem 0 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model  = joblib.load("model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    le     = joblib.load("label_encoder.pkl")
    return model, scaler, le

try:
    model, scaler, le = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info("Make sure model_rf.pkl, scaler.pkl and label_encoder.pkl are in the same folder.")
    models_loaded = False

# ─── Encoding Maps ─────────────────────────────────────────────────────────────
sex_map       = {"Male": "M", "Female": "F"}
sex_encode    = {"F": 0, "M": 1}

chest_map     = {"Typical Angina (ATA)": "ATA",
                 "Asymptomatic (ASY)":   "ASY",
                 "Non-Anginal Pain (NAP)": "NAP",
                 "No Symptoms (TA)":     "TA"}
chest_encode  = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}

ecg_map       = {"Normal": "Normal",
                 "ST-T Wave Abnormality": "ST",
                 "Left Ventricular Hypertrophy": "LVH"}
ecg_encode    = {"LVH": 0, "Normal": 1, "ST": 2}

angina_map    = {"No": "N", "Yes": "Y"}
angina_encode = {"N": 0, "Y": 1}

slope_map     = {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}
slope_encode  = {"Down": 0, "Flat": 1, "Up": 2}

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>❤️ Heart Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict your heart disease risk using Machine Learning</p>",
            unsafe_allow_html=True)
st.markdown("---")

# ─── Input Form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("<p class='section-header'>👤 Personal Info</p>", unsafe_allow_html=True)
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", list(sex_map.keys()))

    st.markdown("<p class='section-header'>🫀 Cardiac Readings</p>", unsafe_allow_html=True)
    chest_pain  = st.selectbox("Chest Pain Type", list(chest_map.keys()))
    resting_bp  = st.number_input("Resting BP (mm Hg)", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=700, value=200)

with col2:
    st.markdown("<p class='section-header'>🩺 Test Results</p>", unsafe_allow_html=True)
    fasting_bs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    resting_ecg     = st.selectbox("Resting ECG", list(ecg_map.keys()))
    max_hr          = st.number_input("Max Heart Rate (MaxHR)", min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Angina", list(angina_map.keys()))

    st.markdown("<p class='section-header'>📈 ST Readings</p>", unsafe_allow_html=True)
    oldpeak  = st.number_input("Oldpeak (ST Depression)", min_value=-5.0, max_value=10.0,
                                value=0.0, step=0.1, format="%.1f")
    st_slope = st.selectbox("ST Slope", list(slope_map.keys()))

st.markdown("<br>", unsafe_allow_html=True)

# ─── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Run Prediction"):
    if not models_loaded:
        st.error("Model files not loaded. Please check your folder.")
    else:
        try:
            sex_val    = sex_encode[sex_map[sex]]
            chest_val  = chest_encode[chest_map[chest_pain]]
            ecg_val    = ecg_encode[ecg_map[resting_ecg]]
            angina_val = angina_encode[angina_map[exercise_angina]]
            slope_val  = slope_encode[slope_map[st_slope]]
            fbs_val    = 1 if fasting_bs == "Yes" else 0

            input_data = pd.DataFrame([{
                "Age":            age,
                "Sex":            sex_val,
                "ChestPainType":  chest_val,
                "RestingBP":      resting_bp,
                "Cholesterol":    cholesterol,
                "FastingBS":      fbs_val,
                "RestingECG":     ecg_val,
                "MaxHR":          max_hr,
                "ExerciseAngina": angina_val,
                "Oldpeak":        oldpeak,
                "ST_Slope":       slope_val,
            }])

            scaled     = scaler.transform(input_data)
            prediction = model.predict(scaled)[0]
            proba      = model.predict_proba(scaled)[0]
            report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if prediction == 1:
                result_text   = "High Risk — Heart Disease Detected"
                result_advice = "Please consult a doctor immediately."
                result_class  = "result-positive"
                confidence    = proba[1] * 100
                emoji         = "⚠️"
            else:
                result_text   = "Low Risk — No Heart Disease Detected"
                result_advice = "Keep maintaining a healthy lifestyle!"
                result_class  = "result-negative"
                confidence    = proba[0] * 100
                emoji         = "✅"

            # Result box
            st.markdown(f"""
            <div class='result-box {result_class}'>
                {emoji} {result_text}<br>
                <small style='font-size:1rem; font-weight:400;'>
                    Confidence: {confidence:.1f}%
                </small>
            </div>""", unsafe_allow_html=True)

            if prediction == 1:
                st.warning(result_advice)
            else:
                st.success(result_advice)

            # ── Report ─────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📄 Your Prediction Report")

            report = f"""
HEART DISEASE PREDICTION REPORT
================================
Date & Time      : {report_date}
Model            : Random Forest Classifier

PATIENT INPUT
-------------
Age              : {age}
Sex              : {sex}
Chest Pain Type  : {chest_pain}
Resting BP       : {resting_bp} mm Hg
Cholesterol      : {cholesterol} mg/dL
Fasting BS >120  : {fasting_bs}
Resting ECG      : {resting_ecg}
Max Heart Rate   : {max_hr}
Exercise Angina  : {exercise_angina}
Oldpeak          : {oldpeak}
ST Slope         : {st_slope}

RESULT
------
Prediction       : {result_text}
Confidence       : {confidence:.1f}%

ADVICE
------
{result_advice}

NOTE: This is an ML-based prediction only.
It is NOT a substitute for professional medical advice.
Always consult a qualified doctor for proper diagnosis.
================================
"""

            # Download button
            st.download_button(
                label="⬇️ Download Report (.txt)",
                data=report,
                file_name=f"heart_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

            # Share / Copy text
            st.markdown("### 📋 Share This Result")
            share_text = (
                f"Heart Disease Prediction — {report_date}\n"
                f"Result     : {result_text}\n"
                f"Confidence : {confidence:.1f}%\n"
                f"Advice     : {result_advice}"
            )
            st.code(share_text, language=None)
            st.caption("Copy the text above to share with your doctor or family.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.exception(e)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Random Forest Model &bull; Heart Disease Dataset &bull; Streamlit App</p>",
    unsafe_allow_html=True
)