import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(current_dir, "Social_Network_Ads.csv")

data = pd.read_csv(csv_path)

X = data[['Age', 'EstimatedSalary']]

y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

model = SVC(kernel='rbf')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

st.title("Customer Purchase Prediction using SVC")

st.sidebar.title("Options")

if st.sidebar.checkbox("Show Dataset"):
    st.subheader("Dataset")
    st.write(data)

if st.sidebar.checkbox("Show Dataset Details"):
    st.subheader("Dataset Information")
    st.write(data.describe())

if st.sidebar.checkbox("Show Heatmap"):
    st.subheader("Correlation Heatmap")

    fig_heat, ax_heat = plt.subplots(figsize=(8,6))

    sns.heatmap(
        data.corr(numeric_only=True),
        annot=True,
        cmap='coolwarm',
        ax=ax_heat
    )

    st.pyplot(fig_heat)

if st.sidebar.checkbox("Show Confusion Matrix"):
    st.subheader("Confusion Matrix")

    fig_cm, ax_cm = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax_cm
    )

    ax_cm.set_xlabel("Predicted")

    ax_cm.set_ylabel("Actual")

    st.pyplot(fig_cm)

st.subheader("Enter Customer Details")

age = st.slider("Age", 18, 60, 30)

salary = st.slider("Estimated Salary", 15000, 150000, 50000)

input_data = np.array([[age, salary]])

input_data = sc.transform(input_data)

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Customer Will Purchase")

    else:
        st.error("Customer Will Not Purchase")

st.subheader("Model Performance")

st.write("Accuracy Score:", round(accuracy, 2))

st.subheader("Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

st.subheader("Final Visualization")

fig_final, ax_final = plt.subplots(figsize=(8,6))

scatter = ax_final.scatter(
    X_test[:,0],
    X_test[:,1],
    c=y_pred
)

ax_final.set_xlabel("Age")

ax_final.set_ylabel("Estimated Salary")

ax_final.set_title("SVC Classification Result")

st.pyplot(fig_final)

