# heart-disease-prediction
ML web app to predict heart disease risk using Random Forest Classifier &amp; Streamlit
# ❤️ Heart Disease Prediction App

> A Machine Learning web application that predicts the risk of heart disease based on patient clinical data — built with Random Forest Classifier and deployed using Streamlit.

---

## 📌 Project Overview

This project uses a **Random Forest Classification** model trained on the [Heart Disease Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) to predict whether a patient is at risk of heart disease. The model takes 11 clinical features as input and returns a prediction along with a confidence score.

A user-friendly **Streamlit web app** allows anyone to enter patient data and instantly get a prediction — no coding required.

---

## 🚀 Features

- 🔍 Predict heart disease risk from clinical inputs
- 📊 Shows prediction confidence percentage
- 📄 Download a full patient report as `.txt`
- 📋 Share result text with doctor or family
- 🎨 Clean, responsive UI built with Streamlit

---

## 🧠 Model Details

| Property        | Value                        |
|----------------|------------------------------|
| Algorithm       | Random Forest Classifier     |
| Preprocessing   | Standard Scaler, Label Encoder |
| Oversampling    | SMOTE (to handle class imbalance) |
| Train/Test Split | 80% / 20%                   |
| Dataset         | heart.csv (918 records)      |

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── appp.py                  # Streamlit web app
├── heart.csv                # Dataset
├── model_rf.pkl             # Trained Random Forest model
├── scaler.pkl               # Fitted Standard Scaler
├── label_encoder.pkl        # Fitted Label Encoder
├── requirements.txt         # Python dependencies
└── Assignment_15_Heart_Disease_Prediction.ipynb  # Full ML notebook
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run appp.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📋 Input Features

| Feature           | Description                              |
|-------------------|------------------------------------------|
| Age               | Age of the patient (years)               |
| Sex               | Male / Female                            |
| Chest Pain Type   | ATA, ASY, NAP, TA                        |
| Resting BP        | Resting blood pressure (mm Hg)           |
| Cholesterol       | Serum cholesterol (mg/dL)                |
| Fasting Blood Sugar | > 120 mg/dL (Yes/No)                  |
| Resting ECG       | Normal, ST, LVH                          |
| Max Heart Rate    | Maximum heart rate achieved              |
| Exercise Angina   | Exercise-induced angina (Yes/No)         |
| Oldpeak           | ST depression induced by exercise        |
| ST Slope          | Slope of peak exercise ST segment        |

---

## 📊 Model Performance

| Metric     | Score  |
|------------|--------|
| Accuracy   | ~90%+  |
| Precision  | High   |
| Recall     | High   |
| F1 Score   | High   |

> Exact scores are available in the Jupyter Notebook.

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Scikit-learn** — Model training & preprocessing
- **Pandas / NumPy** — Data manipulation
- **Imbalanced-learn** — SMOTE oversampling
- **Streamlit** — Web app deployment
- **Joblib** — Model serialization

---

## ⚠️ Disclaimer

This application is built for **educational purposes only**.  
It is **not** a substitute for professional medical advice.  
Always consult a qualified doctor for proper diagnosis and treatment.

---

## 👨‍💻 Author

**Muhammad Muaz**\
Made with ❤️ as part of a Machine Learning Assignment.
