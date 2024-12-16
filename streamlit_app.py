import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import gdown  # مكتبة لتحميل الملفات من Google Drive

# Function to download file from Google Drive
def download_from_google_drive(url, output_path):
    try:
        gdown.download(url, output_path, quiet=False)
    except Exception as e:
        st.error(f"Error downloading the file: {str(e)}")

# Define the log transformation function
def log_transform(data):
    return np.log1p(data)

# Load the trained model
def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
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

    # Google Drive URL of the model
    model_url = 'https://drive.google.com/uc?export=download&id=1Cx91Q_2AlsfidDzktxrCxCM3C3suZQGC'  # Extracted ID from the provided URL
    model_path = 'model.pkl'
    
    # Download the model from Google Drive
    download_from_google_drive(model_url, model_path)

    # Load model
    model = load_model(model_path)

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100, step=1)
        blood_pressure = st.number_input("Blood Pressure Level", min_value=0, max_value=150, value=70, step=1)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)

    with col2:
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80, step=1)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

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
