import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

# Load the pre-trained model
model = joblib.load('decision_tree_model.pkl')

# Sample dataset (you should replace this with your actual dataset)
def load_sample_data():
    data = pd.DataFrame({
        'Pregnancies': [6, 1, 8, 1, 0],
        'Glucose': [148, 85, 183, 89, 137],
        'BloodPressure': [72, 66, 64, 66, 40],
        'SkinThickness': [35, 29, 0, 23, 35],
        'Insulin': [0, 0, 0, 94, 168],
        'BMI': [33.6, 26.6, 23.3, 28.1, 43.1],
        'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288],
        'Age': [50, 31, 32, 21, 33],
        'Outcome': [1, 0, 1, 0, 1]
    })
    return data

def predict_diabetes(input_data):
    """Make diabetes prediction"""
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    prediction_proba = model.predict_proba(input_array)
    
    if prediction[0] == 1:
        return f"Diabetes Detected (Probability: {prediction_proba[0][1]:.2%})"
    else:
        return f"No Diabetes Detected (Probability: {prediction_proba[0][0]:.2%})"

def home_page():
    """Home page with project overview"""
    st.title("Diabetes Risk Prediction Platform")
    
    # Create two columns for image and description
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://cdn.pixabay.com/photo/2016/10/18/18/19/diabetes-1751454_1280.png", use_column_width=True)
    
    with col2:
        st.markdown("""
        ## Project Overview
        
        ### Mission
        Develop an intelligent health screening tool to predict diabetes risk using machine learning.

        ### Key Objectives
        - Early detection of diabetes potential
        - Personalized health risk assessment
        - Encourage preventive healthcare

        ### How It Works
        Our advanced machine learning model analyzes multiple health parameters to estimate diabetes probability, 
        providing insights that can motivate lifestyle and medical interventions.
        """)
    
    # Feature highlights
    st.markdown("### Key Features")
    
    # Create columns for features
    feature_cols = st.columns(3)
    
    features = [
        ("🩺 Comprehensive Analysis", "Evaluates 8 critical health metrics"),
        ("📊 Probability Estimation", "Provides risk percentage, not just binary result"),
        ("🤖 Machine Learning Powered", "Decision Tree algorithm with high accuracy")
    ]
    
    for col, (icon, description) in zip(feature_cols, features):
        with col:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; border: 1px solid #e0e0e0; border-radius: 10px;'>
            <h3>{icon}</h3>
            <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

def visualization_page():
    """Data visualization page"""
    st.title("Diabetes Dataset Insights")
    
    # Load sample data
    data = load_sample_data()
    
    # Visualization options
    viz_type = st.selectbox("Choose Visualization Type", [
        "Correlation Heatmap", 
        "Distribution Plot", 
        "Boxplot by Diabetes Outcome"
    ])
    
    # Create visualizations
    if viz_type == "Correlation Heatmap":
        # Correlation heatmap
        corr_matrix = data.drop('Outcome', axis=1).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        st.pyplot(fig)
    
    elif viz_type == "Distribution Plot":
        # Distribution plot
        feature = st.selectbox("Select Feature", data.columns[:-1])
        fig = px.histogram(data, x=feature, color='Outcome', 
                           marginal='box', 
                           title=f'Distribution of {feature} by Diabetes Outcome')
        st.plotly_chart(fig)
    
    else:
        # Boxplot
        feature = st.selectbox("Select Feature", data.columns[:-1])
        fig = px.box(data, x='Outcome', y=feature, 
                     title=f'{feature} Distribution by Diabetes Outcome')
        st.plotly_chart(fig)
    
    # Dataset summary
    st.subheader("Dataset Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Samples", len(data))
        st.metric("Diabetes Cases", data['Outcome'].sum())
    
    with col2:
        st.metric("Non-Diabetes Cases", len(data) - data['Outcome'].sum())
        st.metric("Diabetes Prevalence", f"{data['Outcome'].mean()*100:.2f}%")

def prediction_page():
    """Prediction page"""
    st.title("Diabetes Risk Prediction")
    
    # Create input columns for better layout
    col1, col2 = st.columns(2)
    
    # Input fields
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)
        
    with col2:
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=79, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0, step=0.1)
    
    # Additional inputs in a third column
    col3, col4 = st.columns(2)
    
    with col3:
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    
    with col4:
        age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
    
    # Prediction button
    if st.button("Predict Diabetes Risk"):
        # Prepare input data in the correct order
        input_data = [
            pregnancies, glucose, blood_pressure, 
            skin_thickness, insulin, bmi, 
            diabetes_pedigree, age
        ]
        
        # Make prediction
        result = predict_diabetes(input_data)
        
        # Display result with custom styling
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
        <h2>🩺 Prediction Result</h2>
        <p style='font-size: 18px;'>{result}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk interpretation section
    st.markdown("### Risk Interpretation Guide")
    risk_guide = {
        "Low Risk (0-30%)": "Consider routine check-ups and preventive lifestyle choices.",
        "Moderate Risk (30-60%)": "Consult healthcare professional. Lifestyle modifications recommended.",
        "High Risk (60-100%)": "Immediate medical consultation advised. Comprehensive health assessment needed."
    }
    
    for risk, advice in risk_guide.items():
        st.markdown(f"- **{risk}**: {advice}")

def main():
    # Set page configuration
    st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")
    
    # Custom CSS for improved styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation
    pages = {
        "🏠 Home": home_page,
        "📊 Data Visualization": visualization_page,
        "🩺 Prediction": prediction_page
    }
    
    # Create navigation
    page = st.sidebar.radio("Navigate", list(pages.keys()))
    
    # Run the selected page
    pages[page]()

if __name__ == "__main__":
    main()
