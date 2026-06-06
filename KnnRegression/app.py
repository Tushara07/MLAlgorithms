import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(current_dir, "Housing.csv")

data = pd.read_csv(csv_path)

X = data[['area', 'bedrooms', 'bathrooms']]

y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

neighbors = range(1, 21)

scores = []

for k in neighbors:

    temp_model = KNeighborsRegressor(n_neighbors=k)

    temp_model.fit(X_train, y_train)

    temp_pred = temp_model.predict(X_test)

    score = r2_score(y_test, temp_pred)

    scores.append(score)

best_k = neighbors[np.argmax(scores)]

model = KNeighborsRegressor(n_neighbors=best_k)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

st.title("House Price Prediction using KNN Regressor")

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

    sns.heatmap(
        data.corr(numeric_only=True),
        annot=True,
        cmap='coolwarm',
        ax=ax_heat
    )

    st.pyplot(fig_heat)

if st.sidebar.checkbox("Show K Value Graph"):
    st.subheader("K Value vs R2 Score")

    fig_k, ax_k = plt.subplots()

    ax_k.plot(neighbors, scores)

    ax_k.set_xlabel("K Value")

    ax_k.set_ylabel("R2 Score")

    ax_k.set_title("K Value vs R2 Score")

    st.pyplot(fig_k)

st.subheader("Enter House Details")

area = st.number_input("Area", 500, 10000, 2000)

bedrooms = st.slider("Bedrooms", 1, 10, 3)

bathrooms = st.slider("Bathrooms", 1, 10, 2)

input_data = np.array([[area, bedrooms, bathrooms]])

input_data = sc.transform(input_data)

if st.button("Predict House Price"):

    prediction = model.predict(input_data)

    st.success(f"Predicted House Price: {prediction[0]:,.2f}")

st.subheader("Model Performance")

st.write("Best K Value:", best_k)

st.write("Mean Squared Error:", round(mse, 2))

st.write("R2 Score:", round(r2, 2))

st.subheader("Final Visualization")

fig_final, ax_final = plt.subplots(figsize=(8,6))

ax_final.scatter(y_test, y_pred)

ax_final.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

ax_final.set_xlabel("Actual Prices")

ax_final.set_ylabel("Predicted Prices")

ax_final.set_title("Actual vs Predicted Prices")

st.pyplot(fig_final)
