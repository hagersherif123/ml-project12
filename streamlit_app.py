import joblib
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# تحميل النموذج
def load_model():
    model = joblib.load('decision_tree_model2.pkl')  # مسار الملف الذي خزنت فيه النموذج
    return model

# تحميل النموذج
model = load_model()

# إضافة تنسيق CSS لتحسين التصميم
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            color: #00008b;
        }
        .stButton>button {
            background-color: #00008b;
            color: white;
            font-size: 16px;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #4682b4;
        }
        .stTitle {
            font-size: 32px;
            font-weight: bold;
            color: #00008b;
        }
        .stSubheader {
            font-size: 22px;
            color: #00008b;
        }
        .stTextInput>div>input {
            background-color: #ffffff;
            border: 2px solid #00008b;
            color: #00008b;
        }
        .stTextInput>div>input:focus {
            border: 2px solid #4682b4;
        }
    </style>
""", unsafe_allow_html=True)

# تطبيق التحويل اللوجاريتمي على المدخلات
def apply_log_transform(input_data):
    return np.log1p(input_data)  # تطبيق log(1+x) بشكل آمن

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

    # إدخال البيانات في مصفوفة
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age])

    # تطبيق التحويل اللوجاريتمي على المدخلات
    input_data_log_transformed = apply_log_transform(input_data)

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

# صفحة Visualization
def visualize_page():
    st.title("Data Visualization")
    st.subheader("Explore important aspects of the dataset")

    # تحميل الداتا
    uploaded_file = st.file_uploader("Choose a dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.write("Dataset Preview")
        st.dataframe(data.head())

        # عرض بعض الرسومات البيانية
        st.subheader("Visualizations")
        
        # رسم المخطط البياني التوضيحي
        if st.checkbox("Show Pairplot"):
            sns.pairplot(data)
            st.pyplot()

        if st.checkbox("Show Correlation Heatmap"):
            plt.figure(figsize=(10, 8))
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
            st.pyplot()

        if st.checkbox("Show Boxplot of Glucose"):
            plt.figure(figsize=(8, 6))
            sns.boxplot(x="Outcome", y="Glucose", data=data)
            st.pyplot()

        if st.checkbox("Show Histogram of Age"):
            plt.figure(figsize=(8, 6))
            sns.histplot(data["Age"], kde=True, bins=20)
            st.pyplot()

# صفحة About
def about_page():
    st.title("About the Project")
    st.image("diabetes_image.jpg", caption="Diabetes Awareness", width=600)
    st.write("""
        This project aims to predict whether a person is diabetic or not based on a dataset that includes various health factors. 
        The dataset primarily focuses on women and includes the following factors: Pregnancies, Glucose, Blood Pressure, 
        Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.
    """)
    
    st.subheader("Dataset Features")
    st.write("""
        1. **Pregnancies**: The number of times the person has been pregnant.
        2. **Glucose**: Plasma glucose concentration in the blood.
        3. **Blood Pressure**: Diastolic blood pressure (mm Hg).
        4. **Skin Thickness**: Triceps skinfold thickness (mm).
        5. **Insulin**: 2-Hour serum insulin (mu U/ml).
        6. **BMI**: Body mass index (weight in kg / height in m^2).
        7. **Diabetes Pedigree Function**: A function that scores the likelihood of diabetes based on family history.
        8. **Age**: The age of the person in years.
    """)

# إعداد شريط التنقل (Navigation Bar)
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["About", "Visualize Data", "Diabetes Prediction"])
    
    if page == "About":
        about_page()
    elif page == "Visualize Data":
        visualize_page()
    elif page == "Diabetes Prediction":
        predict_page()

if __name__ == "__main__":
    main()
