import joblib
import streamlit as st
import numpy as np
import pandas as pd

# تحميل النموذج
def load_model():
    model = joblib.load('decision_tree_model2.pkl')  # مسار الملف الذي خزنت فيه النموذج
    return model

# تحميل النموذج
model = load_model()

# صفحة التنبؤ بمرض السكر
def predict_page():
    st.title("Diabetes Prediction")
    st.subheader("Predict whether a person has diabetes based on input data")

    # إدخال البيانات من المستخدم
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, step=1)
    glucose = st.number_input("Glucose", min_value=44, max_value=199, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=24, max_value=122, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=7, max_value=99, step=1)
    insulin = st.number_input("Insulin", min_value=14, max_value=846, step=1)
    bmi = st.number_input("BMI", min_value=18.2, max_value=67.1, step=0.1)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, step=0.001)
    age = st.number_input("Age", min_value=21, max_value=81, step=1)

    # تطبيق التحويل اللوجاريتمي على المدخلات
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age])

    # استخدام np.log1p لتحويل القيم بطريقة آمنة (log(1 + x))
    input_data_log_transformed = np.log1p(input_data)

    # إعادة تشكيل المصفوفة
    input_data_log_transformed = input_data_log_transformed.reshape(1, -1)

    # التنبؤ بناءً على البيانات المدخلة
    if st.button("Predict"):
        prediction = model.predict(input_data_log_transformed)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        
        # عرض النتيجة مع بعض التنسيق
        st.subheader(f"Prediction: {result}")
        if result == "Diabetic":
            st.write("The model predicts that the person is diabetic.")
        else:
            st.write("The model predicts that the person is not diabetic.")
        
        # عرض ألوان بناءً على النتيجة
        if result == "Diabetic":
            st.markdown("<div style='color:red; font-size:20px;'>Warning: The person might be diabetic. Please consult a doctor.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:green; font-size:20px;'>Good news: The person is not diabetic. Keep it up!</div>", unsafe_allow_html=True)

# صفحة رفع وتحليل الداتا سيت
def dataset_page():
    st.title("Upload and Analyze Dataset")
    st.subheader("Upload a CSV file for diabetes prediction data")

    file = st.file_uploader("Choose a CSV file", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        st.write(data.head())

        st.subheader("Basic Statistics")
        st.write(data.describe())
        
        # تحليل البيانات أو عرض بعض الرسوم البيانية
        st.subheader("Distribution of Glucose Levels")
        st.bar_chart(data['Glucose'].value_counts())

# صفحة شرح المشروع
def about_page():
    st.title("About This Project")
    st.markdown("""
    This is a diabetes prediction app developed using machine learning algorithms. 
    It predicts whether a person is diabetic based on various health metrics.
    
    **Project Features:**
    - Predict diabetes based on input data.
    - Upload and analyze datasets for diabetes prediction.
    
    **Technology Stack:**
    - Machine Learning with Decision Tree
    - Streamlit for Web Interface
    - Python for Data Processing and Analysis
    """)

# إعداد شريط التنقل (Navigation Bar)
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Diabetes Prediction", "Upload Dataset", "About"])
    
    if page == "Diabetes Prediction":
        predict_page()
    elif page == "Upload Dataset":
        dataset_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
