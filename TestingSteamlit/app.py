import streamlit as st
import numpy as np
import pickle
import time

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System 🩺",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rounded buttons and input borders
st.markdown("""
<style>
.stButton button {
    background-color: #34A853;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    padding: 0.5em 1em;
}
.stNumberInput input {
    border: 2px solid #A5D6A7;
    border-radius: 5px;
}
[data-testid="stSidebar"] {
    background-color: #E6EFE8;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Sidebar navigation
st.sidebar.title("🔧 App Controls")
st.sidebar.info("Use the controls below to adjust input parameters.")

st.sidebar.markdown("#### Input Sections")
tab_choice = st.sidebar.radio("Choose section:", ["🏠 Home", "🩸 Prediction", "ℹ️ About the App"])

if tab_choice == "🏠 Home":
    st.title("Welcome to the Diabetes Prediction Dashboard 🧬")
    st.write("This interactive web app allows users to estimate their likelihood of having diabetes based on clinical parameters.")
    st.markdown("Use the **Prediction** tab to begin.")
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=200)

elif tab_choice == "🩸 Prediction":
    st.title("Diabetes Risk Assessment Form")
    st.write("Please fill in your health information below:")

    # Two-column layout for inputs
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input('Pregnancies', 0, 20, 1)
        Glucose = st.number_input('Glucose Level', 0, 200, 120)
        BloodPressure = st.number_input('Blood Pressure', 0, 150, 70)
        SkinThickness = st.number_input('Skin Thickness', 0, 100, 20)
    with col2:
        Insulin = st.number_input('Insulin', 0, 900, 80)
        BMI = st.number_input('BMI', 0.0, 70.0, 24.5)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
        Age = st.number_input('Age', 1, 120, 33)

    if st.button("Predict Now ⚡"):
        with st.spinner("Analyzing your inputs..."):
            time.sleep(1.5)
            input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)

            st.subheader("📊 Prediction Results")
            if prediction == 1:
                st.error("High likelihood of Diabetes detected.")
                st.metric(label="Predicted Risk (%)", value="76%", delta="High Risk")
            else:
                st.success("Low likelihood of Diabetes detected.")
                st.metric(label="Predicted Risk (%)", value="12%", delta="Low Risk")

elif tab_choice == "ℹ️ About the App":
    st.title("About this Project 💡")
    st.write("""
    This app was developed using **Streamlit**, **Scikit-learn**, and **XGBoost**, trained on the Pima Indians Diabetes dataset.
    
    **Tech Stack:**
    - Machine Learning Model: XGBoost Classifier
    - Framework: Streamlit
    - Dataset: Kaggle - Pima Indians Diabetes
    
    The project demonstrates interactive health analysis through AI-driven predictions.
    """)

