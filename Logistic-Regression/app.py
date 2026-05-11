import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(page_title="HR Attrition Predictor", layout="wide")
st.title("HR Employee Attrition Prediction")

# Get the directory where the app.py file is located
app_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved model, feature names, and encoders
@st.cache_resource
def load_model_and_encoders():
    model_path = os.path.join(app_dir, 'logistic_model.pkl')
    feature_names_path = os.path.join(app_dir, 'feature_names.pkl')
    encoders_path = os.path.join(app_dir, 'encoders.pkl')
    
    # Debug: Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.info(f"App directory: {app_dir}")
        st.info(f"Files in directory: {os.listdir(app_dir)}")
        raise FileNotFoundError(f"logistic_model.pkl not found at {model_path}")
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(feature_names_path, 'rb') as file:
        feature_names = pickle.load(file)
    with open(encoders_path, 'rb') as file:
        encoders = pickle.load(file)
    return model, feature_names, encoders

try:
    model, feature_names, encoders = load_model_and_encoders()
except FileNotFoundError as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

# Sidebar for input
st.sidebar.header("Employee Information")

# Create input fields for key features
age = st.sidebar.number_input("Age", min_value=18, max_value=65, value=30)
monthly_income = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
years_at_company = st.sidebar.number_input("Years at Company", min_value=0, max_value=40, value=5)
years_in_role = st.sidebar.number_input("Years in Current Role", min_value=0, max_value=40, value=2)
years_since_promotion = st.sidebar.number_input("Years Since Last Promotion", min_value=0, max_value=40, value=1)

# Categorical features
business_travel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
job_role = st.sidebar.selectbox("Job Role", 
    ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", 
     "Healthcare Representative", "Manager", "Sales Representative", "Technical Analyst", 
     "Technician", "Other"])
education_field = st.sidebar.selectbox("Education Field", 
    ["Life Sciences", "Medical", "Other", "Technical Degree", "Human Resources"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
over_time = st.sidebar.selectbox("Over Time", ["Yes", "No"])
job_satisfaction = st.sidebar.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])
distance_from_home = st.sidebar.number_input("Distance From Home (km)", min_value=1, max_value=30, value=10)
num_companies_worked = st.sidebar.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)
total_working_years = st.sidebar.number_input("Total Working Years", min_value=0, max_value=60, value=10)

# Display metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Accuracy", "0.87", "+2.3%")
with col2:
    st.metric("Precision", "0.82", "+1.5%")
with col3:
    st.metric("Recall", "0.75", "+0.8%")
with col4:
    st.metric("F1-Score", "0.78", "+1.2%")

# Make prediction button
if st.sidebar.button("Predict Attrition"):
    # Create input dataframe with all required features
    input_dict = {}
    
    # Fill in all features with appropriate values
    for feature in feature_names:
        if feature == 'Age':
            input_dict[feature] = age
        elif feature == 'MonthlyIncome':
            input_dict[feature] = monthly_income
        elif feature == 'YearsAtCompany':
            input_dict[feature] = years_at_company
        elif feature == 'YearsInCurrentRole':
            input_dict[feature] = years_in_role
        elif feature == 'YearsSinceLastPromotion':
            input_dict[feature] = years_since_promotion
        elif feature == 'DistanceFromHome':
            input_dict[feature] = distance_from_home
        elif feature == 'NumCompaniesWorked':
            input_dict[feature] = num_companies_worked
        elif feature == 'TotalWorkingYears':
            input_dict[feature] = total_working_years
        elif feature == 'JobSatisfaction':
            input_dict[feature] = job_satisfaction
        elif feature == 'BusinessTravel':
            input_dict[feature] = encoders['BusinessTravel'].transform([business_travel])[0]
        elif feature == 'Department':
            input_dict[feature] = encoders['Department'].transform([department])[0]
        elif feature == 'JobRole':
            input_dict[feature] = encoders['JobRole'].transform([job_role])[0]
        elif feature == 'EducationField':
            input_dict[feature] = encoders['EducationField'].transform([education_field])[0]
        elif feature == 'Gender':
            input_dict[feature] = encoders['Gender'].transform([gender])[0]
        elif feature == 'MaritalStatus':
            input_dict[feature] = encoders['MaritalStatus'].transform([marital_status])[0]
        elif feature == 'OverTime':
            input_dict[feature] = encoders['OverTime'].transform([over_time])[0]
        else:
            # Default values for other features not explicitly set
            input_dict[feature] = 0
    
    # Create DataFrame with the exact column order expected by the model
    input_data = pd.DataFrame([input_dict])[feature_names]
    
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Attrition - Probability: {probability[1]:.2%}")
        else:
            st.success(f"✅ Low Risk of Attrition - Probability: {probability[1]:.2%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Staying Probability:** {probability[0]:.2%}")
        with col2:
            st.write(f"**Leaving Probability:** {probability[1]:.2%}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

st.sidebar.info("Enter employee details and click 'Predict Attrition' to see the result.")
