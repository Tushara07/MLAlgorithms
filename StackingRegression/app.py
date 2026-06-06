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
    "stacking_model.pkl"
)

model = joblib.load(model_path)

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(
    current_dir,
    "laptop_price.csv"
)

data = pd.read_csv(
    csv_path,
    encoding="latin1"
)

st.title("Laptop Price Prediction using Stacking Regressor")

company_list = [
    'Acer', 'Apple', 'Asus', 'Chuwi', 'Dell',
    'Fujitsu', 'Google', 'HP', 'Huawei', 'LG',
    'Lenovo', 'MSI', 'Mediacom', 'Microsoft',
    'Razer', 'Samsung', 'Toshiba', 'Vero', 'Xiaomi'
]

typename_list = [
    '2 in 1 Convertible',
    'Gaming',
    'Netbook',
    'Notebook',
    'Ultrabook',
    'Workstation'
]

os_list = [
    'Android',
    'Chrome OS',
    'Linux',
    'Mac OS X',
    'No OS',
    'Windows 10',
    'Windows 10 S',
    'Windows 7',
    'macOS'
]

st.sidebar.title("Options")

if st.sidebar.checkbox("Show Dataset"):

    st.subheader("Dataset")

    st.write(data)

if st.sidebar.checkbox("Show Dataset Details"):

    st.subheader("Dataset Information")

    st.write(data.describe())

if st.sidebar.checkbox("Show Price Distribution"):

    st.subheader("Laptop Price Distribution")

    fig_dist, ax_dist = plt.subplots(figsize=(8,6))

    sns.histplot(
        data['Price_euros'],
        bins=30,
        ax=ax_dist
    )

    ax_dist.set_title(
        "Price Distribution"
    )

    st.pyplot(fig_dist)


if st.sidebar.checkbox("Show Model Comparison"):

    scores = {
        "Linear Regression": 0.6100,
        "Decision Tree": 0.7873,
        "Random Forest": 0.8038,
        "Stacking": 0.8145
    }

    st.subheader(
        "Model Performance Comparison"
    )

    fig_comp, ax_comp = plt.subplots(
        figsize=(8,5)
    )

    ax_comp.bar(
        scores.keys(),
        scores.values()
    )

    ax_comp.set_ylabel(
        "R2 Score"
    )

    ax_comp.set_title(
        "Base Learners vs Stacking"
    )

    plt.xticks(rotation=15)

    st.pyplot(fig_comp)

st.subheader("Enter Laptop Details")

with st.form("prediction_form"):

    company = st.selectbox(
        "Company",
        company_list
    )

    typename = st.selectbox(
        "Laptop Type",
        typename_list
    )

    inches = st.number_input(
        "Screen Size (Inches)",
        min_value=10.0,
        max_value=20.0,
        value=15.6
    )

    ram = st.selectbox(
        "RAM (GB)",
        [2, 4, 8, 16, 32, 64]
    )

    os = st.selectbox(
        "Operating System",
        os_list
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=0.5,
        max_value=5.0,
        value=2.0
    )

    submit = st.form_submit_button(
        "Predict Laptop Price"
    )

company_map = {
    'Acer':0,
    'Apple':1,
    'Asus':2,
    'Chuwi':3,
    'Dell':4,
    'Fujitsu':5,
    'Google':6,
    'HP':7,
    'Huawei':8,
    'LG':9,
    'Lenovo':10,
    'MSI':11,
    'Mediacom':12,
    'Microsoft':13,
    'Razer':14,
    'Samsung':15,
    'Toshiba':16,
    'Vero':17,
    'Xiaomi':18
}

type_map = {
    '2 in 1 Convertible':0,
    'Gaming':1,
    'Netbook':2,
    'Notebook':3,
    'Ultrabook':4,
    'Workstation':5
}

os_map = {
    'Android':0,
    'Chrome OS':1,
    'Linux':2,
    'Mac OS X':3,
    'No OS':4,
    'Windows 10':5,
    'Windows 10 S':6,
    'Windows 7':7,
    'macOS':8
}

if submit:

    input_data = np.array([[
        company_map[company],
        type_map[typename],
        inches,
        ram,
        os_map[os],
        weight
    ]])

    prediction = model.predict(
        input_data
    )

    st.success(
        f"Predicted Laptop Price: €{prediction[0]:.2f}"
    )

st.subheader("Model Performance")

st.write(
    "Linear Regression R²:",
    0.6100
)

st.write(
    "Decision Tree R²:",
    0.7873
)

st.write(
    "Random Forest R²:",
    0.8038
)

st.write(
    "Stacking Regressor R²:",
    0.8145
)

st.subheader("Project Architecture")

st.write(
    "Base Learners: Linear Regression, Decision Tree, Random Forest"
)

st.write(
    "Meta Learner: Linear Regression"
)

