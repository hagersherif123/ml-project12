import requests
import pickle
import streamlit as st

# Load the model directly from Google Drive
def load_model_from_google_drive(url):
    try:
        # Get the model file from Google Drive
        response = requests.get(url)
        if response.status_code == 200:
            with open('model.pkl', 'wb') as f:
                f.write(response.content)
            
            # Load the model using pickle
            with open('model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        else:
            st.error("Failed to download the model.")
            return None
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

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
    model_url = "https://drive.google.com/uc?export=download&id=1lYJ9OJk2Z5hK713PrJM1k0us9siSPdNo"
    model = load_model_from_google_drive(model_url)

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

if __name__ == "__main__":
    main()
