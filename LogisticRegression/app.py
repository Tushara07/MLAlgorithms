import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(current_dir, "diabetes.csv")

data = pd.read_csv(csv_path)

X = data.drop('Outcome', axis=1)

y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.title("Diabetes Prediction using Logistic Regression")

st.sidebar.title("Options")

if st.sidebar.checkbox("Show Dataset"):
    st.subheader("Dataset")
    st.write(data)

if st.sidebar.checkbox("Show Dataset Details"):
    st.subheader("Dataset Information")
    st.write(data.describe())

if st.sidebar.checkbox("Show Heatmap"):
    st.subheader("Correlation Heatmap")

    fig_heat, ax_heat = plt.subplots(figsize=(10,8))

    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax_heat)

    st.pyplot(fig_heat)

if st.sidebar.checkbox("Show Confusion Matrix"):
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig_cm, ax_cm = plt.subplots()

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)

    ax_cm.set_xlabel("Predicted")

    ax_cm.set_ylabel("Actual")

    st.pyplot(fig_cm)

st.subheader("Enter Patient Details")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)

glucose = st.number_input("Glucose", 0, 200, 100)

blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)

skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

insulin = st.number_input("Insulin", 0, 900, 80)

bmi = st.number_input("BMI", 0.0, 70.0, 25.0)

dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

age = st.number_input("Age", 1, 100, 30)

input_data = np.array([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
]])

input_data = sc.transform(input_data)

if st.button("Predict"):

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("The person is Diabetic")

    else:
        st.success("The person is Not Diabetic")

st.subheader("Model Accuracy")

st.write("Accuracy Score:", round(accuracy, 2))

st.subheader("Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)
