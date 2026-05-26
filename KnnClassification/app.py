import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(current_dir, "loan.csv")

data = pd.read_csv(csv_path)

data['Gender'] = data['Gender'].fillna(
    data['Gender'].mode()[0]
)

data['Married'] = data['Married'].fillna(
    data['Married'].mode()[0]
)

data['Dependents'] = data['Dependents'].fillna(
    data['Dependents'].mode()[0]
)

data['Self_Employed'] = data['Self_Employed'].fillna(
    data['Self_Employed'].mode()[0]
)

data['LoanAmount'] = data['LoanAmount'].fillna(
    data['LoanAmount'].median()
)

data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(
    data['Loan_Amount_Term'].median()
)

data['Credit_History'] = data['Credit_History'].fillna(
    data['Credit_History'].mode()[0]
)

le = LabelEncoder()

columns = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Property_Area',
    'Loan_Status'
]

for col in columns:

    data[col] = le.fit_transform(data[col])

X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)

y = data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

neighbors = range(1, 21)

scores = []

for k in neighbors:

    temp_model = KNeighborsClassifier(n_neighbors=k)

    temp_model.fit(X_train, y_train)

    temp_pred = temp_model.predict(X_test)

    score = accuracy_score(y_test, temp_pred)

    scores.append(score)

best_k = neighbors[np.argmax(scores)]

model = KNeighborsClassifier(
    n_neighbors=best_k,
    weights='distance'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

st.title("Loan Approval Prediction using KNN Classifier")

st.sidebar.title("Options")

if st.sidebar.checkbox("Show Dataset"):
    st.subheader("Dataset")
    st.write(data)

if st.sidebar.checkbox("Show Dataset Details"):
    st.subheader("Dataset Information")
    st.write(data.describe())

if st.sidebar.checkbox("Show Heatmap"):
    st.subheader("Correlation Heatmap")

    fig_heat, ax_heat = plt.subplots(figsize=(12,8))

    sns.heatmap(
        data.corr(numeric_only=True),
        annot=True,
        cmap='coolwarm',
        ax=ax_heat
    )

    st.pyplot(fig_heat)

if st.sidebar.checkbox("Show K Value Graph"):

    st.subheader("K Value vs Accuracy")

    fig_k, ax_k = plt.subplots()

    ax_k.plot(neighbors, scores)

    ax_k.set_xlabel("K Value")

    ax_k.set_ylabel("Accuracy")

    ax_k.set_title("K Value vs Accuracy")

    st.pyplot(fig_k)

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

st.subheader("Enter Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])

married = st.selectbox("Married", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

applicant_income = st.number_input(
    "Applicant Income",
    0,
    100000,
    5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    0,
    50000,
    2000
)

loan_amount = st.number_input(
    "Loan Amount",
    0,
    1000,
    150
)

loan_term = st.number_input(
    "Loan Amount Term",
    0,
    500,
    360
)

credit_history = st.selectbox(
    "Credit History",
    [1.0, 0.0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

gender_val = 1 if gender == "Male" else 0

married_val = 1 if married == "Yes" else 0

education_val = 0 if education == "Graduate" else 1

self_employed_val = 1 if self_employed == "Yes" else 0

property_map = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}

property_val = property_map[property_area]

dependents_map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}

dependents_val = dependents_map[dependents]

input_data = np.array([[
    gender_val,
    married_val,
    dependents_val,
    education_val,
    self_employed_val,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_val
]])

input_data = sc.transform(input_data)

if st.button("Predict Loan Status"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    st.write(
        "Approval Probability:",
        round(probability[0][1] * 100, 2),
        "%"
    )

    if prediction[0] == 1:
        st.success("Loan Approved")

    else:
        st.error("Loan Rejected")
        

st.subheader("Model Performance")

st.write("Best K Value:", best_k)

st.write("Accuracy Score:", round(accuracy, 2))

st.subheader("Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

