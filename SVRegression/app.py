import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(current_dir, "insurance.csv")

data = pd.read_csv(csv_path)

data = data.drop_duplicates()

le = LabelEncoder()

data['sex'] = le.fit_transform(data['sex'])

data['smoker'] = le.fit_transform(data['smoker'])

data['region'] = le.fit_transform(data['region'])

X = data.drop('charges', axis=1)

y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

y_train = sc_y.fit_transform(y_train.values.reshape(-1,1))

model = SVR(kernel='rbf')

model.fit(X_train, y_train.ravel())

y_pred = model.predict(X_test)

y_pred_original = sc_y.inverse_transform(y_pred.reshape(-1,1))

mse = mean_squared_error(y_test, y_pred_original)

r2 = r2_score(y_test, y_pred_original)

st.title("Medical Insurance Cost Prediction using SVM Regression")

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

    sns.heatmap(data.corr(numeric_only=True),
                annot=True,
                cmap='coolwarm',
                ax=ax_heat)

    st.pyplot(fig_heat)

if st.sidebar.checkbox("Actual vs Predicted Graph"):
    st.subheader("Actual vs Predicted Charges")

    fig_graph, ax_graph = plt.subplots()

    ax_graph.scatter(y_test, y_pred_original)

    ax_graph.set_xlabel("Actual Charges")

    ax_graph.set_ylabel("Predicted Charges")

    ax_graph.set_title("Actual vs Predicted Charges")

    st.pyplot(fig_graph)

st.subheader("Enter Patient Details")

age = st.slider("Age", 18, 100, 30)

sex = st.selectbox("Sex", ["Male", "Female"])

bmi = st.slider("BMI", 10.0, 50.0, 25.0)

children = st.slider("Children", 0, 10, 1)

smoker = st.selectbox("Smoker", ["Yes", "No"])

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

sex_value = 1 if sex == "Male" else 0

smoker_value = 1 if smoker == "Yes" else 0

region_mapping = {
    "southwest": 3,
    "southeast": 2,
    "northwest": 1,
    "northeast": 0
}

region_value = region_mapping[region]

input_data = np.array([[
    age,
    sex_value,
    bmi,
    children,
    smoker_value,
    region_value
]])

input_data = sc_X.transform(input_data)

if st.button("Predict Insurance Charges"):

    prediction = model.predict(input_data)

    prediction = sc_y.inverse_transform(
        prediction.reshape(-1,1)
    )

    st.success(
        f"Predicted Insurance Charges: ${prediction[0][0]:.2f}"
    )

st.subheader("Model Performance")

st.write("Mean Squared Error:", round(mse, 2))

st.write("R2 Score:", round(r2, 2))

st.subheader("Final Visualization")

fig_final, ax_final = plt.subplots(figsize=(8,6))

ax_final.scatter(y_test, y_pred_original)

ax_final.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

ax_final.set_xlabel("Actual Charges")

ax_final.set_ylabel("Predicted Charges")

ax_final.set_title("Actual vs Predicted Insurance Charges")

st.pyplot(fig_final)