import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import requests

# Define the log transformation function
def log_transform(data):
    return np.log1p(data)

# Load the trained model
def load_model(file_url):
    try:
        model_file = requests.get(file_url)
        if model_file.status_code == 200:
            with open("/tmp/decision_tree_model.pkl", "wb") as f:
                f.write(model_file.content)

            with open("/tmp/decision_tree_model.pkl", 'rb') as file:
                model = pickle.load(file)
            return model
        else:
            st.error("Error downloading the file from Google Drive.")
            return None
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Preprocess the input data
def preprocess_input(data):
    # Apply log transformation to the necessary columns
    columns_to_transform = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
    for column in columns_to_transform:
        if column in data:
            data[column] = log_transform(data[column])  # Apply log transform
    return data

# Load and clean the dataset
def load_dataset(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Visualize correlations
def visualize_correlations(df):
    correlation = df.corr()
    fig = px.imshow(correlation, text_auto=True, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")
    st.title("🩺 Diabetes Prediction App")
    st.write("Enter the patient's details below to predict the likelihood of diabetes.")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0.0, max_value=17.0, value=1.0, step=1.0)
        glucose = st.number_input("Glucose Level", min_value=44.0, max_value=199.0, value=100.0, step=1.0)
        blood_pressure = st.number_input("Blood Pressure Level", min_value=24.0, max_value=122.0, value=70.0, step=1.0)
        skin_thickness = st.number_input("Skin Thickness", min_value=7.0, max_value=99.0, value=20.0, step=1.0)

    with col2:
        insulin = st.number_input("Insulin Level", min_value=14.0, max_value=846.0, value=80.0, step=1.0)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=18.2, max_value=67.1, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=21.0, max_value=81.0, value=30.0, step=1.0)

    # Load model from Google Drive
    model_url = 'https://drive.google.com/file/d/1lYJ9OJk2Z5hK713PrJM1k0us9siSPdNo/view?usp=sharing'  # Use direct download link
    model = load_model(model_url)

    if model is not None:
        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }

        # Preprocess input data (apply log transformation)
        input_data = preprocess_input(input_data)

        input_df = pd.DataFrame(input_data, index=[0])

        try:
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            # Display prediction
            st.markdown(f"""
                <div style="font-size: 24px; padding: 10px; background-color: #f0f4f8; border: 2px solid #3e9f7d; border-radius: 5px; text-align: center;">
                    <strong>Prediction:</strong> {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}
                </div>
            """, unsafe_allow_html=True)

            # Display probability
            st.markdown(f"""
                <div style="font-size: 20px; padding: 10px; background-color: #e8f5e9; border: 2px solid #4caf50; border-radius: 5px; text-align: center;">
                    <strong>Probability of being Diabetic:</strong> {prediction_proba[0][1]:.2f}
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Load the dataset and preprocess it for visualization
    dataset_file = st.file_uploader("Upload a CSV file containing diabetes data 📂", type="csv")
    if dataset_file is not None:
        df = load_dataset(dataset_file)
        if df is not None:
            # Apply log transformation to the dataset as well
            df = preprocess_input(df)

            # Display visualizations
            visualize_correlations(df)

if __name__ == "__main__":
    main()
