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
    st.subheader("Upload your dataset and predict whether a person has diabetes or not")

    # تحميل الداتا
    uploaded_file = st.file_uploader("Choose a dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.write("Dataset Preview")
        st.dataframe(data.head())

        # الأعمدة التي سيتم استخدام التحويل اللوجاريتمي عليها
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].apply(lambda x: np.log1p(x))

        # عرض بعض الإحصائيات
        st.subheader("Statistics")
        st.write(data.describe())

        # التنبؤ بناءً على البيانات المدخلة
        if st.button("Predict"):
            # تأكد من أن الأعمدة المطلوبة موجودة في البيانات
            required_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            
            # التأكد من أن جميع الأعمدة المطلوبة موجودة في البيانات
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                # إزالة عمود Outcome (إذا كان موجودًا)
                X = data[required_cols]
                X_log_transformed = X.apply(np.log1p)  # تطبيق التحويل اللوجاريتمي على البيانات المدخلة

                # التنبؤ باستخدام النموذج
                prediction = model.predict(X_log_transformed)
                result = ["Diabetic" if pred == 1 else "Non-Diabetic" for pred in prediction]

                # إضافة التنبؤ إلى البيانات
                data["Prediction"] = result  
                st.subheader("Predictions")
                st.dataframe(data.head())

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
