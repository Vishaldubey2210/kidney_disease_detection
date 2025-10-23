import streamlit as st
import numpy as np
import pickle
import os

# Model loading - ensure 'models/best_rf_model.pkl' exists
MODEL_PATH = 'models/best_rf_model.pkl'
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at '{MODEL_PATH}'. Please save your trained model first.")
    st.stop()

model = pickle.load(open(MODEL_PATH, 'rb'))

st.title('Kidney Disease Prediction App')

# Features list (order must match training)
features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 
            'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# User inputs
input_data = []
input_data.append(st.number_input('Age', min_value=1, max_value=120, value=30))
input_data.append(st.number_input('Blood Pressure', min_value=0, max_value=200, value=80))
input_data.append(st.selectbox('Specific Gravity', [1.005, 1.010, 1.015, 1.020, 1.025]))
input_data.append(st.selectbox('Albumin', [0,1,2,3,4,5]))
input_data.append(st.selectbox('Sugar', [0,1,2,3,4,5]))
input_data.append(st.number_input('Blood Glucose Random', min_value=0, max_value=500, value=120))
input_data.append(st.number_input('Blood Urea', min_value=0, max_value=300, value=40))
input_data.append(st.number_input('Serum Creatinine', min_value=0.0, max_value=20.0, value=1.0, format="%.2f"))
input_data.append(st.number_input('Sodium', min_value=0, max_value=200, value=140))
input_data.append(st.number_input('Potassium', min_value=0.0, max_value=20.0, value=4.0, format="%.2f"))
input_data.append(st.number_input('Hemoglobin', min_value=0.0, max_value=30.0, value=13.5))
input_data.append(st.number_input('Packed Cell Volume', min_value=0, max_value=100, value=40))
input_data.append(st.number_input('White Blood Cell Count', min_value=0, max_value=20000, value=8000))
input_data.append(st.number_input('Red Blood Cell Count', min_value=0.0, max_value=10.0, value=5.0, format="%.2f"))
input_data.append(st.selectbox('Hypertension (0=No, 1=Yes)', [0,1]))
input_data.append(st.selectbox('Diabetes Mellitus (0=No, 1=Yes)', [0,1]))
input_data.append(st.selectbox('Coronary Artery Disease (0=No, 1=Yes)', [0,1]))
input_data.append(st.selectbox('Appetite (0=Normal, 1=Poor)', [0,1]))
input_data.append(st.selectbox('Pedal Edema (0=No, 1=Yes)', [0,1]))
input_data.append(st.selectbox('Anemia (0=No, 1=Yes)', [0,1]))

# Convert inputs to array
input_data = np.array([input_data])

# Prediction on button click
if st.button('Predict'):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)

    if prediction[0] == 0:
        st.error(f"Prediction: High risk of Chronic Kidney Disease.\nProbability: {proba[0][0]:.2f}")
    else:
        st.success(f"Prediction: Low risk of Chronic Kidney Disease.\nProbability: {proba[0][1]:.2f}")
