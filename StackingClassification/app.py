import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

current_dir = os.path.dirname(__file__)

model_path = os.path.join(
    current_dir,
    "stacking_classifier.pkl"
)

csv_path = os.path.join(
    current_dir,
    "train.csv"
)

model = joblib.load(model_path)

data = pd.read_csv(csv_path)

st.title("Mobile Price Range Classification using Stacking Classifier")

st.sidebar.title("Options")

if st.sidebar.checkbox("Show Dataset"):

    st.subheader("Dataset")

    st.write(data)

if st.sidebar.checkbox("Show Dataset Details"):

    st.subheader("Dataset Information")

    st.write(data.describe())

if st.sidebar.checkbox("Show Price Range Distribution"):

    st.subheader("Price Range Distribution")

    fig_dist, ax_dist = plt.subplots(figsize=(8,6))

    sns.countplot(
        x='price_range',
        data=data,
        ax=ax_dist
    )

    ax_dist.set_title(
        "Mobile Price Range Distribution"
    )

    st.pyplot(fig_dist)

if st.sidebar.checkbox("Show Correlation Heatmap"):

    st.subheader("Correlation Heatmap")

    fig_heat, ax_heat = plt.subplots(figsize=(12,8))

    sns.heatmap(
        data.corr(),
        cmap='coolwarm',
        ax=ax_heat,
        annot=True
    )

    st.pyplot(fig_heat)

if st.sidebar.checkbox("Show Model Comparison"):

    st.subheader("Model Performance Comparison")

    models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Stacking"
    ]

    scores = [
        0.8875,
        0.8400,
        0.9350,
        0.9075
    ]

    fig_comp, ax_comp = plt.subplots(figsize=(8,5))

    ax_comp.bar(
        models,
        scores
    )

    ax_comp.set_ylabel(
        "Accuracy"
    )

    ax_comp.set_title(
        "Model Comparison"
    )

    for i, score in enumerate(scores):

        ax_comp.text(
            i,
            score + 0.005,
            f"{score:.4f}",
            ha='center'
        )

    st.pyplot(fig_comp)

st.subheader("Enter Mobile Specifications")

with st.form("prediction_form"):

    battery_power = st.number_input(
        "Battery Power (mAh)",
        min_value=500,
        max_value=2500,
        value=1500
    )

    ram = st.number_input(
        "RAM (MB)",
        min_value=256,
        max_value=8000,
        value=3000
    )

    px_height = st.number_input(
        "Pixel Height",
        min_value=0,
        max_value=3000,
        value=1000
    )

    px_width = st.number_input(
        "Pixel Width",
        min_value=0,
        max_value=3000,
        value=1500
    )

    mobile_wt = st.number_input(
        "Mobile Weight (g)",
        min_value=80,
        max_value=250,
        value=150
    )

    n_cores = st.slider(
        "Number of Cores",
        1,
        8,
        4
    )

    submit = st.form_submit_button(
        "Predict Price Range"
    )

if submit:

    input_data = np.array([[
        battery_power,
        ram,
        px_height,
        px_width,
        mobile_wt,
        n_cores
    ]])

    prediction = model.predict(
        input_data
    )[0]

    price_ranges = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }

    st.success(
        f"Predicted Price Range: {price_ranges[prediction]}"
    )

st.subheader("Model Performance")

st.write(
    "Logistic Regression Accuracy:",
    "88.75%"
)

st.write(
    "Decision Tree Accuracy:",
    "84.00%"
)

st.write(
    "Random Forest Accuracy:",
    "93.50%"
)

st.write(
    "Stacking Classifier Accuracy:",
    "90.75%"
)

st.subheader("Project Architecture")

st.write(
    "Base Learners: Logistic Regression, Decision Tree, Random Forest"
)

st.write(
    "Meta Learner: Logistic Regression"
)

