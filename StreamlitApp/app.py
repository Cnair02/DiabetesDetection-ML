import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Diabetes Prediction App")

st.write("Enter patient details to predict the likelihood of diabetes.")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
blood_pressure = st.number_input("Blood Pressure", 0, 150)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 800)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    model = pickle.load(open("./diabetes_model.pkl", "rb"))
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(features)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.success(f"Prediction: {result}")
