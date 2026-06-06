import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import tree

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(current_dir, "gld_price_data.csv")

data = pd.read_csv(csv_path)

X = data.drop(['Date', 'GLD'], axis=1)

y = data['GLD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

depths = range(1, 21)

scores = []

for d in depths:

    temp_model = DecisionTreeRegressor(
        max_depth=d,
        random_state=42
    )

    temp_model.fit(X_train, y_train)

    temp_pred = temp_model.predict(X_test)

    score = r2_score(y_test, temp_pred)

    scores.append(score)

best_depth = depths[np.argmax(scores)]

model = DecisionTreeRegressor(
    max_depth=best_depth,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

st.title("Gold Price Prediction using Decision Tree Regressor")

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

if st.sidebar.checkbox("Show Depth Graph"):

    st.subheader("Max Depth vs R2 Score")

    fig_depth, ax_depth = plt.subplots()

    ax_depth.plot(depths, scores)

    ax_depth.set_xlabel("Max Depth")

    ax_depth.set_ylabel("R2 Score")

    ax_depth.set_title("Max Depth vs R2 Score")

    st.pyplot(fig_depth)

if st.sidebar.checkbox("Show Feature Importance"):

    st.subheader("Feature Importance")

    importance = model.feature_importances_

    features = X.columns

    fig_imp, ax_imp = plt.subplots(figsize=(8,6))

    ax_imp.bar(features, importance)

    ax_imp.set_xlabel("Features")

    ax_imp.set_ylabel("Importance")

    ax_imp.set_title("Feature Importance")

    st.pyplot(fig_imp)

if st.sidebar.checkbox("Show Decision Tree"):

    st.subheader("Decision Tree Visualization")

    fig_tree, ax_tree = plt.subplots(figsize=(18,10))

    tree.plot_tree(
        model,
        feature_names=X.columns,
        filled=True,
        max_depth=3,
        fontsize=8,
        ax=ax_tree
    )

    st.pyplot(fig_tree)

st.subheader("Enter Market Details")

spx = st.number_input("SPX", 500.0, 5000.0, 1500.0)

uso = st.number_input("USO", 0.0, 100.0, 50.0)

slv = st.number_input("SLV", 0.0, 100.0, 20.0)

eur_usd = st.number_input("EUR/USD", 0.5, 2.0, 1.2)

input_data = np.array([[spx, uso, slv, eur_usd]])

if st.button("Predict Gold Price"):

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Gold Price: {prediction[0]:.2f}"
    )

st.subheader("Model Performance")

st.write("Best Max Depth:", best_depth)

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

ax_final.set_xlabel("Actual Gold Prices")

ax_final.set_ylabel("Predicted Gold Prices")

ax_final.set_title("Actual vs Predicted Gold Prices")

st.pyplot(fig_final)

