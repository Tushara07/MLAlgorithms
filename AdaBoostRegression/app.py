import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(
    current_dir,
    "CAR DETAILS FROM CAR DEKHO.csv"
)

data = pd.read_csv(csv_path)

data = data.drop_duplicates()

Q1 = data['selling_price'].quantile(0.25)

Q3 = data['selling_price'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

data = data[
    (data['selling_price'] >= lower) &
    (data['selling_price'] <= upper)
]

le = LabelEncoder()

columns = [
    'fuel',
    'seller_type',
    'transmission',
    'owner'
]

for col in columns:

    data[col] = le.fit_transform(data[col])

X = data[
    [
        'year',
        'km_driven',
        'fuel',
        'seller_type',
        'transmission',
        'owner'
    ]
]

y = data['selling_price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

estimators = [10, 50, 100, 150]

scores = []

for n in estimators:

    temp_model = AdaBoostRegressor(
        n_estimators=n,
        random_state=42
    )

    temp_model.fit(X_train, y_train)

    temp_pred = temp_model.predict(X_test)

    score = r2_score(y_test, temp_pred)

    scores.append(score)

best_n = estimators[np.argmax(scores)]

model = AdaBoostRegressor(
    n_estimators=best_n,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

st.title("Car Price Prediction using AdaBoost Regressor")

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

if st.sidebar.checkbox("Show Price Distribution"):

    st.subheader("Selling Price Distribution")

    fig_dist, ax_dist = plt.subplots()

    sns.histplot(
        data['selling_price'],
        bins=50,
        ax=ax_dist
    )

    st.pyplot(fig_dist)

if st.sidebar.checkbox("Show Estimator Graph"):

    st.subheader("Estimators vs R2 Score")

    fig_est, ax_est = plt.subplots()

    ax_est.plot(estimators, scores)

    ax_est.set_xlabel("Estimators")

    ax_est.set_ylabel("R2 Score")

    ax_est.set_title(
        "Estimators vs R2 Score"
    )

    st.pyplot(fig_est)

if st.sidebar.checkbox("Show Feature Importance"):

    st.subheader("Feature Importance")

    importance = model.feature_importances_

    features = X.columns

    fig_imp, ax_imp = plt.subplots(
        figsize=(8,6)
    )

    ax_imp.barh(features, importance)

    ax_imp.set_xlabel("Importance")

    ax_imp.set_ylabel("Features")

    ax_imp.set_title(
        "Feature Importance"
    )

    st.pyplot(fig_imp)

st.subheader("Enter Car Details")

with st.form("prediction_form"):

    year = st.slider(
        "Year",
        2000,
        2024,
        2018
    )

    km_driven = st.number_input(
        "Kilometers Driven",
        0,
        500000,
        50000
    )

    fuel = st.selectbox(
        "Fuel Type",
        [
            "Petrol",
            "Diesel",
            "CNG",
            "LPG"
        ]
    )

    seller_type = st.selectbox(
        "Seller Type",
        [
            "Dealer",
            "Individual",
            "Trustmark Dealer"
        ]
    )

    transmission = st.selectbox(
        "Transmission",
        [
            "Manual",
            "Automatic"
        ]
    )

    owner = st.selectbox(
        "Owner",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner"
        ]
    )

    submit = st.form_submit_button(
        "Predict Car Price"
    )

fuel_map = {
    "CNG": 0,
    "Diesel": 1,
    "LPG": 2,
    "Petrol": 3
}

seller_map = {
    "Dealer": 0,
    "Individual": 1,
    "Trustmark Dealer": 2
}

transmission_map = {
    "Automatic": 0,
    "Manual": 1
}

owner_map = {
    "First Owner": 0,
    "Second Owner": 1,
    "Third Owner": 2,
    "Fourth & Above Owner": 3
}

input_data = np.array([[
    year,
    km_driven,
    fuel_map[fuel],
    seller_map[seller_type],
    transmission_map[transmission],
    owner_map[owner]
]])

if submit:

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Car Price: ₹{prediction[0]:,.2f}"
    )

st.subheader("Model Performance")

st.write(
    "Best Number of Estimators:",
    best_n
)

st.write(
    "Mean Squared Error:",
    round(mse, 2)
)

st.write(
    "R2 Score:",
    round(r2, 2)
)

st.subheader("Final Visualization")

fig_final, ax_final = plt.subplots(
    figsize=(8,6)
)

ax_final.scatter(
    y_test,
    y_pred
)

ax_final.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

ax_final.set_xlabel(
    "Actual Prices"
)

ax_final.set_ylabel(
    "Predicted Prices"
)

ax_final.set_title(
    "Actual vs Predicted Car Prices"
)

st.pyplot(fig_final)

