import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(current_dir, "CustomerChurn.csv")

data = pd.read_csv(csv_path)

data['TotalCharges'] = pd.to_numeric(
    data['TotalCharges'],
    errors='coerce'
)

data['TotalCharges'] = data['TotalCharges'].fillna(
    data['TotalCharges'].median()
)

le = LabelEncoder()

columns = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'Churn'
]

for col in columns:

    data[col] = le.fit_transform(data[col])

X = data.drop(['customerID', 'Churn'], axis=1)

y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

depths = range(1, 21)

scores = []

for d in depths:

    temp_model = DecisionTreeClassifier(
        max_depth=d,
        random_state=42
    )

    temp_model.fit(X_train, y_train)

    temp_pred = temp_model.predict(X_test)

    score = accuracy_score(y_test, temp_pred)

    scores.append(score)

best_depth = depths[np.argmax(scores)]

model = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

st.title("Customer Churn Prediction using Decision Tree Classifier")

st.sidebar.title("Options")

if st.sidebar.checkbox("Show Dataset"):

    st.subheader("Dataset")

    st.write(data)

if st.sidebar.checkbox("Show Dataset Details"):

    st.subheader("Dataset Information")

    st.write(data.describe())

if st.sidebar.checkbox("Show Heatmap"):

    st.subheader("Correlation Heatmap")

    fig_heat, ax_heat = plt.subplots(figsize=(14,10))

    sns.heatmap(
        data.corr(numeric_only=True),
        cmap='coolwarm',
        annot=True,
        ax=ax_heat
    )

    st.pyplot(fig_heat)

if st.sidebar.checkbox("Show Depth Graph"):

    st.subheader("Max Depth vs Accuracy")

    fig_depth, ax_depth = plt.subplots()

    ax_depth.plot(depths, scores)

    ax_depth.set_xlabel("Max Depth")

    ax_depth.set_ylabel("Accuracy")

    ax_depth.set_title("Max Depth vs Accuracy")

    st.pyplot(fig_depth)

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

if st.sidebar.checkbox("Show Feature Importance"):

    st.subheader("Feature Importance")

    importance = model.feature_importances_

    features = X.columns

    fig_imp, ax_imp = plt.subplots(figsize=(10,6))

    ax_imp.barh(features, importance)

    ax_imp.set_xlabel("Importance")

    ax_imp.set_ylabel("Features")

    ax_imp.set_title("Feature Importance")

    st.pyplot(fig_imp)

if st.sidebar.checkbox("Show Decision Tree"):

    st.subheader("Decision Tree Visualization")

    fig_tree, ax_tree = plt.subplots(figsize=(20,10))

    tree.plot_tree(
        model,
        feature_names=X.columns,
        filled=True,
        max_depth=3,
        fontsize=8,
        ax=ax_tree
    )

    st.pyplot(fig_tree)

st.subheader("Enter Customer Details")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.slider(
    "Tenure",
    0,
    72,
    12
)

monthly = st.number_input(
    "Monthly Charges",
    0.0,
    200.0,
    70.0
)

total = st.number_input(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

gender_val = 1 if gender == "Male" else 0

senior_val = 1 if senior == "Yes" else 0

partner_val = 1 if partner == "Yes" else 0

dependents_val = 1 if dependents == "Yes" else 0

paperless_val = 1 if paperless == "Yes" else 0

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

contract_val = contract_map[contract]

input_data = np.array([[
    gender_val,
    senior_val,
    partner_val,
    dependents_val,
    tenure,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    contract_val,
    paperless_val,
    2,
    monthly,
    total,
    0
]])

if st.button("Predict Churn"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    st.write(
        "Churn Probability:",
        round(probability[0][1] * 100, 2),
        "%"
    )

    if prediction[0] == 1:

        st.error("Customer Will Churn")

    else:

        st.success("Customer Will Stay")

st.subheader("Model Performance")

st.write("Best Max Depth:", best_depth)

st.write("Accuracy Score:", round(accuracy, 2))

st.subheader("Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

