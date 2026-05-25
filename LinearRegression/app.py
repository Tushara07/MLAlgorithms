import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import os

current_dir = os.path.dirname(__file__)

st.write("Current Directory:", current_dir)
st.write("Files:", os.listdir(current_dir))

csv_path = os.path.join(current_dir, "advertising.csv")

data = pd.read_csv(csv_path)

X = data[['TV']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

st.title("Sales Prediction using Linear Regression")

st.sidebar.title("Options")

if st.sidebar.checkbox("Show Dataset"):
    st.subheader("Dataset")
    st.write(data)

if st.sidebar.checkbox("Show Dataset Details"):
    st.subheader("Dataset Information")
    st.write(data.describe())

if st.sidebar.checkbox("Show Heatmap"):
    st.subheader("Correlation Heatmap")

    fig_heat, ax_heat = plt.subplots()

    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax_heat)

    st.pyplot(fig_heat)

st.subheader("Sales Prediction")

tv = st.slider("Enter TV Advertising Budget", 0, 300, 100)

prediction = model.predict([[tv]])

st.success(f"Predicted Sales: {prediction[0]:.2f}")

st.subheader("Model Accuracy")

st.write("R2 Score:", round(score, 2))

X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

X_line_df = pd.DataFrame(X_line, columns=['TV'])

y_line = model.predict(X_line_df)

fig, ax = plt.subplots()

ax.scatter(X, y)

ax.plot(X_line, y_line)

ax.scatter(tv, prediction[0], s=100)

ax.set_xlabel("TV")
ax.set_ylabel("Sales")
ax.set_title("Linear Regression: TV vs Sales")

st.pyplot(fig)

