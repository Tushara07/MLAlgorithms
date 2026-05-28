import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(current_dir, "breast-cancer.csv")

data = pd.read_csv(csv_path)

le = LabelEncoder()

data['diagnosis'] = le.fit_transform(
    data['diagnosis']
)

X = data.drop(['id', 'diagnosis'], axis=1)

y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

estimators = [10, 50, 100, 150]

scores = []

for n in estimators:

    temp_model = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    temp_model.fit(X_train, y_train)

    temp_pred = temp_model.predict(X_test)

    score = accuracy_score(y_test, temp_pred)

    scores.append(score)

best_n = estimators[np.argmax(scores)]

model = RandomForestClassifier(
    n_estimators=best_n,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

st.title("Breast Cancer Prediction using Random Forest Classifier")

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
        cmap='coolwarm',
        annot=True,
        ax=ax_heat
    )

    st.pyplot(fig_heat)

if st.sidebar.checkbox("Show Diagnosis Count"):

    st.subheader("Diagnosis Count")

    fig_count, ax_count = plt.subplots()

    sns.countplot(
        x='diagnosis',
        data=data,
        ax=ax_count
    )

    st.pyplot(fig_count)

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

if st.sidebar.checkbox("Show Estimator Graph"):

    st.subheader("Estimators vs Accuracy")

    fig_est, ax_est = plt.subplots()

    ax_est.plot(estimators, scores)

    ax_est.set_xlabel("Number of Estimators")

    ax_est.set_ylabel("Accuracy")

    ax_est.set_title("Estimators vs Accuracy")

    st.pyplot(fig_est)

if st.sidebar.checkbox("Show Feature Importance"):

    st.subheader("Feature Importance")

    importance = model.feature_importances_

    features = X.columns

    fig_imp, ax_imp = plt.subplots(figsize=(10,8))

    ax_imp.barh(features, importance)

    ax_imp.set_xlabel("Importance")

    ax_imp.set_ylabel("Features")

    ax_imp.set_title("Feature Importance")

    st.pyplot(fig_imp)

if st.sidebar.checkbox("Show ROC Curve"):

    st.subheader("ROC Curve")

    y_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )

    auc_score = roc_auc_score(
        y_test,
        y_prob
    )

    fig_roc, ax_roc = plt.subplots()

    ax_roc.plot(fpr, tpr)

    ax_roc.plot([0,1], [0,1], 'r--')

    ax_roc.set_xlabel(
        "False Positive Rate"
    )

    ax_roc.set_ylabel(
        "True Positive Rate"
    )

    ax_roc.set_title(
        f"ROC Curve (AUC = {auc_score:.2f})"
    )

    st.pyplot(fig_roc)

st.subheader("Enter Tumor Details")

with st.form("prediction_form"):

    radius_mean = st.number_input(
        "Radius Mean",
        value=14.0
    )

    texture_mean = st.number_input(
        "Texture Mean",
        value=19.0
    )

    perimeter_mean = st.number_input(
        "Perimeter Mean",
        value=90.0
    )

    area_mean = st.number_input(
        "Area Mean",
        value=650.0
    )

    smoothness_mean = st.number_input(
        "Smoothness Mean",
        value=0.09
    )

    submit = st.form_submit_button(
        "Predict Diagnosis"
    )

input_data = np.zeros((1, X.shape[1]))

input_data[0][0] = radius_mean
input_data[0][1] = texture_mean
input_data[0][2] = perimeter_mean
input_data[0][3] = area_mean
input_data[0][4] = smoothness_mean

if submit:

    prediction = model.predict(input_data)

    probability = model.predict_proba(
        input_data
    )

    cancer_probability = round(
        probability[0][1] * 100,
        2
    )

    st.write(
        f"Cancer Probability: {cancer_probability}%"
    )

    if prediction[0] == 1:

        st.error(
            "Prediction: Malignant Tumor"
        )

    else:

        st.success(
            "Prediction: Benign Tumor"
        )

st.subheader("Model Performance")

st.write(
    "Best Number of Estimators:",
    best_n
)

st.write(
    "Accuracy Score:",
    round(accuracy, 2)
)

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred
)

st.text(report)
