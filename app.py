import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Sidebar project info
with st.sidebar:
    st.markdown("## ❤️ Heart Disease Prediction")
    st.write("by Shivam Pandey")

st.title("Heart Disease Prediction App")
st.markdown(
    "Upload your heart disease dataset CSV below (columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target), enter patient parameters, and click Predict."
)

# --- 1. UPLOAD CSV WIDGET ---
uploaded_file = st.file_uploader("Upload your Heart Disease CSV data", type=["csv"], help="CSV must include columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target")

def load_model(data):
    try:
        # Ensure binary output for UCI heart data
        data['target'] = (data['target'] > 0).astype(int)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        X = data.drop('target', axis=1)
        y = data['target']
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        y = y.dropna()
        X = X[:len(y)]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=5000, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
        return model, X_train.shape[1], imputer, scaler
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
        return None, None, None, None

# --- 2. RUN MODEL TRAINING IF CSV IS UPLOADED ---
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    model, num_features, imputer, scaler = load_model(data)
else:
    st.info("Please upload a CSV file to use the app.")
    st.stop()

# --- 3. PATIENT INPUT FORM ---
col1, col2 = st.columns(2)
with col1:
    st.header("📋 Patient Information (Part 1)")
    age = st.slider('Age (years)', 20, 80, 50)
    sex = st.selectbox('Sex', ['Female (0)', 'Male (1)'])
    sex = int(sex.split('(')[1].strip(')'))
    cp = st.slider('Chest Pain Type (0-3)', 0, 3, 0)
    trestbps = st.slider('Resting Blood Pressure (mmHg)', 90, 200, 120)
    chol = st.slider('Cholesterol (mg/dl)', 100, 400, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120?', ['No (0)', 'Yes (1)'])
    fbs = int(fbs.split('(')[1].strip(')'))
    restecg = st.slider('Resting ECG Results (0-2)', 0, 2, 0)

with col2:
    st.header("📋 Patient Information (Part 2)")
    thalach = st.slider('Maximum Heart Rate (thalach, bpm)', 50, 210, 150)
    exang = st.selectbox('Exercise Induced Angina?', ['No (0)', 'Yes (1)'])
    exang = int(exang.split('(')[1].strip(')'))
    oldpeak = st.slider('ST Depression (0.0-6.0)', 0.0, 6.0, 1.0)
    slope = st.slider('Slope of Peak Exercise ST (0-2)', 0, 2, 1)
    ca = st.slider('Major Vessels Colored (ca, 0-3)', 0, 3, 0)
    thal = st.slider('Thalassemia (thal, 0-3)', 0, 3, 0)

# --- 4. PREDICT BUTTON ---
st.markdown("---")
if st.button('🔍 Predict Heart Disease Risk', use_container_width=True):
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                            thalach, exang, oldpeak, slope, ca, thal]])
    user_input = imputer.transform(user_input)
    user_input = scaler.transform(user_input)
    st.write(f"🔍 Your input has {user_input.shape[1]} features")
    st.write(f"📊 Model expects {num_features} features")
    if user_input.shape[1] != num_features:
        st.error(f"❌ ERROR: Input has {user_input.shape[1]} features but model expects {num_features}!")
    else:
        probability = model.predict_proba(user_input)[0]
        st.markdown("---")
        st.header("🏥 Prediction Result")
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            if probability[1] > 0.7:
                st.error("⚠️ HEART DISEASE LIKELY DETECTED")
                st.write(f"**Risk Level: {probability[1]*100:.2f}%**")
                st.write("⚠️ Please consult a medical professional.")
            elif probability[1] > 0.4:
                st.warning("Risk Borderline: Consider professional advice.")
                st.write(f"**Risk Level: {probability[1]*100:.2f}%**")
            else:
                st.success("✅ NO HEART DISEASE DETECTED")
                st.write(f"**Healthy Confidence: {probability[0]*100:.2f}%**")
        with result_col2:
            st.write("**Probability Breakdown:**")
            st.write(f"- No Disease: {probability[0]*100:.2f}%")
            st.write(f"- Disease: {probability[1]*100:.2f}%")
            chart_data = pd.DataFrame(
                {'Probability': [probability[0]*100, probability[1]*100]},
                index=['No Disease', 'Disease']
            )
            st.bar_chart(chart_data)
st.markdown("---")
st.markdown("**Disclaimer:** For educational purposes only. Consult a healthcare professional for proper medical advice.")
