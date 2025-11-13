import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Configure the page
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Add title
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter your health information and click 'Predict' to get results.")

# IMPORTANT: Use @st.cache_resource to train model only ONCE
@st.cache_resource
def load_model():
    """Load and train the model (runs only once)"""
    try:
        # Load data
        data = pd.read_csv(r"C:/Users/BIRENDRA/OneDrive/Desktop/heart-disease-app/data/heart.csv")
        
        st.write(f"✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        st.write(f"📊 Column names: {list(data.columns)}")
        
        # Convert any text columns to numbers
        data = data.replace('Male', 1)
        data = data.replace('Female', 0)
        data = data.replace('male', 1)
        data = data.replace('female', 0)
        
        # Convert all columns to numeric
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        st.write(f"NaN values before imputation: {data.isnull().sum().sum()}")
        
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # IMPORTANT: Use SimpleImputer to handle NaN values properly
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        st.write(f"NaN values after imputation: {pd.DataFrame(X).isnull().sum().sum()}")
        st.write(f"✅ Dataset ready: {X.shape[0]} rows, {X.shape[1]} features")
        st.write(f"📊 Feature count: {X.shape[1]}")
        
        # Remove NaN from y as well
        y = y.dropna()
        
        # Make sure X and y have same length
        X = X[:len(y)]
        
        st.write(f"✅ Final dataset: {len(y)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.write(f"✅ Training set: {X_train.shape[0]} samples")
        st.write(f"✅ Testing set: {X_test.shape[0]} samples")
        
        # Train model
        model = LogisticRegression(max_iter=5000, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
        st.write(f"📊 Model expects {X_train.shape[1]} features for prediction")
        
        # IMPORTANT: Save feature count for later prediction
        return model, X_train.shape[1], imputer
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
        return None, None, None

# Load the model
model, num_features, imputer = load_model()

if model is not None:
    # Create two columns
    col1, col2 = st.columns(2)

    # Column 1 - First set of inputs
    with col1:
        st.header("📋 Patient Information (Part 1)")
        age = st.slider('Age (years)', 20, 80, 50)
        sex = st.selectbox('Sex', ['Female (0)', 'Male (1)'])
        sex = int(sex.split('(')[1].strip(')'))
        cp = st.slider('Chest Pain Type (0-3)', 0, 3, 0)
        trestbps = st.slider('Resting Blood Pressure (mmHg)', 90, 200, 120)
        cholestoral = st.slider('Cholesterol (mg/dl)', 100, 400, 200)
        fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120?', ['No (0)', 'Yes (1)'])
        fasting_blood_sugar = int(fasting_blood_sugar.split('(')[1].strip(')'))
        rest_ecg = st.slider('Resting ECG Results (0-2)', 0, 2, 0)

    # Column 2 - Second set of inputs
    with col2:
        st.header("📋 Patient Information (Part 2)")
        Max_heart_rate = st.slider('Maximum Heart Rate (bpm)', 50, 210, 150)
        exercise_induced_angina = st.selectbox('Exercise Induced Angina?', ['No (0)', 'Yes (1)'])
        exercise_induced_angina = int(exercise_induced_angina.split('(')[1].strip(')'))
        oldpeak = st.slider('ST Depression (0.0-6.0)', 0.0, 6.0, 1.0)
        slope = st.slider('Slope of Peak Exercise ST (0-2)', 0, 2, 1)
        vessels_colored_by_flouro = st.slider('Major Vessels Colored (0-3)', 0, 3, 0)
        thalassemia = st.slider('Thalassemia (0-3)', 0, 3, 0)

    # Predict button
    st.markdown("---")
    if st.button('🔍 Predict Heart Disease Risk', use_container_width=True):
        # Create input array with EXACT number of features the model expects
        user_input = np.array([[age, sex, cp, trestbps, cholestoral, fasting_blood_sugar, rest_ecg, 
                               Max_heart_rate, exercise_induced_angina, oldpeak, slope, vessels_colored_by_flouro, thalassemia]])
        
        st.write(f"🔍 Your input has {user_input.shape[1]} features")
        st.write(f"📊 Model expects {num_features} features")
        
        # IMPORTANT: Make sure feature count matches
        if user_input.shape[1] != num_features:
            st.error(f"❌ ERROR: Input has {user_input.shape[1]} features but model expects {num_features}!")
            st.write("This usually means your CSV has different columns than expected.")
        else:
            # Get prediction
            prediction = model.predict(user_input)[0]
            probability = model.predict_proba(user_input)[0]
            
            # Display results
            st.markdown("---")
            st.header("🏥 Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ HEART DISEASE LIKELY DETECTED")
                    st.write(f"**Risk Level: {probability[1]*100:.2f}%**")
                    st.write("⚠️ Please consult a medical professional.")
                else:
                    st.success("✅ NO HEART DISEASE DETECTED")
                    st.write(f"**Healthy Confidence: {probability[0]*100:.2f}%**")
            
            with col2:
                st.write("**Probability Breakdown:**")
                st.write(f"- No Disease: {probability[0]*100:.2f}%")
                st.write(f"- Disease: {probability[1]*100:.2f}%")
                
                # Show as bar chart
                chart_data = pd.DataFrame(
                    {'Probability': [probability[0]*100, probability[1]*100]},
                    index=['No Disease', 'Disease']
                )
                st.bar_chart(chart_data)

    st.markdown("---")
    st.markdown("**Disclaimer:** For educational purposes only. Consult a healthcare professional for proper medical advice.")
else:
    st.error("❌ Model could not be loaded. Streamlit will retry when you save changes.")
