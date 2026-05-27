import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(current_dir, "AB_NYC_2019.csv")

data = pd.read_csv(csv_path)

data['name'] = data['name'].fillna("Unknown")

data['host_name'] = data['host_name'].fillna("Unknown")

data['last_review'] = data['last_review'].fillna(
    "Not Available"
)

data['reviews_per_month'] = data[
    'reviews_per_month'
].fillna(
    data['reviews_per_month'].median()
)

Q1 = data['price'].quantile(0.25)

Q3 = data['price'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

data = data[
    (data['price'] >= lower) &
    (data['price'] <= upper)
]

le = LabelEncoder()

columns = [
    'neighbourhood_group',
    'neighbourhood',
    'room_type'
]

for col in columns:

    data[col] = le.fit_transform(data[col])

X = data[[
    'neighbourhood_group',
    'room_type',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'availability_365'
]]

y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

estimators = [10, 50, 100, 150]

scores = []

for n in estimators:

    temp_model = RandomForestRegressor(
        n_estimators=n,
        random_state=42
    )

    temp_model.fit(X_train, y_train)

    temp_pred = temp_model.predict(X_test)

    score = r2_score(y_test, temp_pred)

    scores.append(score)

best_n = estimators[np.argmax(scores)]

model = RandomForestRegressor(
    n_estimators=best_n,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

st.title("Airbnb Price Prediction using Random Forest Regressor")

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
        ax=ax_heat
    )

    st.pyplot(fig_heat)

if st.sidebar.checkbox("Show Price Distribution"):

    st.subheader("Price Distribution")

    fig_dist, ax_dist = plt.subplots(figsize=(8,6))

    sns.histplot(data['price'], bins=50, ax=ax_dist)

    ax_dist.set_title("Price Distribution")

    st.pyplot(fig_dist)

if st.sidebar.checkbox("Show Estimator Graph"):

    st.subheader("Estimators vs R2 Score")

    fig_est, ax_est = plt.subplots()

    ax_est.plot(estimators, scores)

    ax_est.set_xlabel("Number of Estimators")

    ax_est.set_ylabel("R2 Score")

    ax_est.set_title("Estimators vs R2 Score")

    st.pyplot(fig_est)

if st.sidebar.checkbox("Show Feature Importance"):

    st.subheader("Feature Importance")

    importance = model.feature_importances_

    features = X.columns

    fig_imp, ax_imp = plt.subplots(figsize=(8,6))

    ax_imp.barh(features, importance)

    ax_imp.set_xlabel("Importance")

    ax_imp.set_ylabel("Features")

    ax_imp.set_title("Feature Importance")

    st.pyplot(fig_imp)

st.subheader("Enter Airbnb Details")

with st.form("prediction_form"):

    neighbourhood_group = st.selectbox(
        "Neighbourhood Group",
        [
            "Brooklyn",
            "Manhattan",
            "Queens",
            "Bronx",
            "Staten Island"
        ]
    )

    room_type = st.selectbox(
        "Room Type",
        [
            "Entire home/apt",
            "Private room",
            "Shared room"
        ]
    )

    minimum_nights = st.slider(
        "Minimum Nights",
        1,
        30,
        2
    )

    number_of_reviews = st.slider(
        "Number of Reviews",
        0,
        500,
        10
    )

    reviews_per_month = st.slider(
        "Reviews Per Month",
        0.0,
        20.0,
        1.0
    )

    availability_365 = st.slider(
        "Availability 365",
        0,
        365,
        100
    )

    submit = st.form_submit_button(
        "Predict Airbnb Price"
    )

group_map = {
    "Bronx": 0,
    "Brooklyn": 1,
    "Manhattan": 2,
    "Queens": 3,
    "Staten Island": 4
}

room_map = {
    "Entire home/apt": 0,
    "Private room": 1,
    "Shared room": 2
}

group_val = group_map[neighbourhood_group]

room_val = room_map[room_type]

input_data = np.array([[
    group_val,
    room_val,
    minimum_nights,
    number_of_reviews,
    reviews_per_month,
    availability_365
]])

if submit:

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Airbnb Price: ${prediction[0]:.2f}"
    )

st.subheader("Model Performance")

st.write("Best Number of Estimators:", best_n)

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

ax_final.set_title(
    "Actual vs Predicted Airbnb Prices"
)

st.pyplot(fig_final)
