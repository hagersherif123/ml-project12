import joblib
import streamlit as st
import numpy as np

# تحميل النموذج
def load_model():
    model = joblib.load('decision_tree_model2.pkl')  # مسار الملف الذي خزنت فيه النموذج
    return model

# تحميل النموذج
model = load_model()

# واجهة المستخدم في Streamlit
st.title("Diabetes Prediction")

# إدخال البيانات من المستخدم
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17)
glucose = st.number_input("Glucose", min_value=44, max_value=199)
blood_pressure = st.number_input("Blood Pressure", min_value=24, max_value=122)
skin_thickness = st.number_input("Skin Thickness", min_value=7, max_value=99)
insulin = st.number_input("Insulin", min_value=14, max_value=846)
bmi = st.number_input("BMI", min_value=18.2, max_value=67.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42)
age = st.number_input("Age", min_value=21, max_value=81)

# إنشاء مصفوفة البيانات المدخلة
input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)

# التنبؤ بناءً على البيانات المدخلة
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Prediction: ", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
