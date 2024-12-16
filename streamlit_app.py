import pickle
import requests
from io import BytesIO
import streamlit as st

# تحميل الملف من Google Drive
def download_file_from_drive(file_url):
    file_id = file_url.split('/d/')[1].split('/')[0]
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    response = requests.get(download_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        st.error("Error downloading the file from Google Drive.")
        return None

# تحميل الموديل
def load_model(file_url):
    try:
        model_file = download_file_from_drive(file_url)
        if model_file:
            model = pickle.load(model_file)
            return model
        else:
            st.error("Failed to download the model file.")
            return None
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# تحميل log_transform
def load_log_transform(file_url):
    try:
        log_transform_file = download_file_from_drive(file_url)
        if log_transform_file:
            log_transform = pickle.load(log_transform_file)
            return log_transform
        else:
            st.error("Failed to download log transform file.")
            return None
    except Exception as e:
        st.error(f"Error loading the log transform: {str(e)}")
        return None

# Main Streamlit app
def main():
    st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")
    st.title("🩺 Diabetes Prediction App")
    st.write("Enter the patient's details below to predict the likelihood of diabetes.")

    # Define the Google Drive links (with direct download)
    model_url = "https://drive.google.com/file/d/1Cx91Q_2AlsfidDzktxrCxCM3C3suZQGC/view?usp=sharing"
    log_transform_url = "https://drive.google.com/file/d/1URz0ERV8mycKj9M2JABAUHNfq7qVyRxo/view?usp=sharing"

    # تحميل الموديل و log_transform
    model = load_model(model_url)
    log_transform = load_log_transform(log_transform_url)

    if model is not None and log_transform is not None:
        # Your code for prediction
        pass

if __name__ == "__main__":
    main()
