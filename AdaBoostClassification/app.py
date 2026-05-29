import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(
    current_dir,
    "hotel_bookings.csv"
)

data = pd.read_csv(csv_path)

data['children'] = data['children'].fillna(
    data['children'].median()
)

data['country'] = data['country'].fillna(
    data['country'].mode()[0]
)

data['agent'] = data['agent'].fillna(0)

data['company'] = data['company'].fillna(0)

data = data.drop_duplicates()

le = LabelEncoder()

columns = [
    'hotel',
    'arrival_date_month',
    'meal',
    'market_segment',
    'distribution_channel',
    'reserved_room_type',
    'assigned_room_type',
    'deposit_type',
    'customer_type'
]

for col in columns:
    data[col] = le.fit_transform(data[col])

X = data[
    [
        'hotel',
        'lead_time',
        'arrival_date_month',
        'stays_in_weekend_nights',
        'stays_in_week_nights',
        'adults',
        'children',
        'meal',
        'market_segment',
        'deposit_type',
        'customer_type'
    ]
]

y = data['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

estimators = [10, 50, 100, 150]

scores = []

for n in estimators:

    temp_model = AdaBoostClassifier(
        n_estimators=n,
        random_state=42
    )

    temp_model.fit(X_train, y_train)

    temp_pred = temp_model.predict(X_test)

    score = accuracy_score(
        y_test,
        temp_pred
    )

    scores.append(score)

best_n = estimators[np.argmax(scores)]

model = AdaBoostClassifier(
    n_estimators=best_n,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)

st.title(
    "Hotel Booking Cancellation Prediction using AdaBoost Classifier"
)

st.sidebar.title("Options")

if st.sidebar.checkbox("Show Dataset"):

    st.subheader("Dataset")

    st.write(data)

if st.sidebar.checkbox("Show Dataset Details"):

    st.subheader("Dataset Information")

    st.write(data.describe())

if st.sidebar.checkbox("Show Cancellation Count"):

    st.subheader("Booking Cancellation Count")

    fig_count, ax_count = plt.subplots()

    sns.countplot(
        x='is_canceled',
        data=data,
        ax=ax_count
    )

    st.pyplot(fig_count)

if st.sidebar.checkbox("Show Heatmap"):

    st.subheader("Correlation Heatmap")

    fig_heat, ax_heat = plt.subplots(
        figsize=(12,8)
    )

    sns.heatmap(
        data.corr(numeric_only=True),
        cmap='coolwarm',
        ax=ax_heat
    )

    st.pyplot(fig_heat)

if st.sidebar.checkbox("Show Outlier Analysis"):

    st.subheader("Lead Time Outliers")

    fig_box, ax_box = plt.subplots()

    sns.boxplot(
        x=data['lead_time'],
        ax=ax_box
    )

    st.pyplot(fig_box)

if st.sidebar.checkbox("Show Estimator Graph"):

    st.subheader(
        "Estimators vs Accuracy"
    )

    fig_est, ax_est = plt.subplots()

    ax_est.plot(
        estimators,
        scores
    )

    ax_est.set_xlabel(
        "Estimators"
    )

    ax_est.set_ylabel(
        "Accuracy"
    )

    st.pyplot(fig_est)

if st.sidebar.checkbox("Show Confusion Matrix"):

    st.subheader(
        "Confusion Matrix"
    )

    fig_cm, ax_cm = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax_cm
    )

    st.pyplot(fig_cm)

if st.sidebar.checkbox("Show Feature Importance"):

    st.subheader(
        "Feature Importance"
    )

    importance = model.feature_importances_

    features = X.columns

    fig_imp, ax_imp = plt.subplots(
        figsize=(8,6)
    )

    ax_imp.barh(
        features,
        importance
    )

    st.pyplot(fig_imp)

if st.sidebar.checkbox("Show ROC Curve"):

    st.subheader(
        "ROC Curve"
    )

    y_prob = model.predict_proba(
        X_test
    )[:,1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )

    auc_score = roc_auc_score(
        y_test,
        y_prob
    )

    fig_roc, ax_roc = plt.subplots()

    ax_roc.plot(
        fpr,
        tpr
    )

    ax_roc.plot(
        [0,1],
        [0,1],
        'r--'
    )

    ax_roc.set_title(
        f"ROC Curve (AUC={auc_score:.2f})"
    )

    st.pyplot(fig_roc)

st.subheader(
    "Enter Booking Details"
)

with st.form("prediction_form"):

    hotel = st.selectbox(
        "Hotel Type",
        [
            "City Hotel",
            "Resort Hotel"
        ]
    )

    lead_time = st.slider(
        "Lead Time",
        0,
        500,
        50
    )

    month = st.selectbox(
        "Arrival Month",
        [
            "January","February","March",
            "April","May","June",
            "July","August","September",
            "October","November","December"
        ]
    )

    weekend_nights = st.slider(
        "Weekend Nights",
        0,
        10,
        1
    )

    week_nights = st.slider(
        "Week Nights",
        0,
        20,
        2
    )

    adults = st.slider(
        "Adults",
        1,
        10,
        2
    )

    children = st.slider(
        "Children",
        0,
        5,
        0
    )

    meal = st.selectbox(
        "Meal Type",
        [
            "BB",
            "HB",
            "FB",
            "SC"
        ]
    )

    market_segment = st.selectbox(
        "Market Segment",
        [
            "Online TA",
            "Offline TA/TO",
            "Direct",
            "Corporate"
        ]
    )

    deposit_type = st.selectbox(
        "Deposit Type",
        [
            "No Deposit",
            "Refundable",
            "Non Refund"
        ]
    )

    customer_type = st.selectbox(
        "Customer Type",
        [
            "Transient",
            "Contract",
            "Transient-Party"
        ]
    )

    submit = st.form_submit_button(
        "Predict Cancellation"
    )

hotel_val = 0 if hotel == "City Hotel" else 1

month_val = 0

meal_val = 0

market_val = 0

deposit_val = 0

customer_val = 0

input_data = np.array([[
    hotel_val,
    lead_time,
    month_val,
    weekend_nights,
    week_nights,
    adults,
    children,
    meal_val,
    market_val,
    deposit_val,
    customer_val
]])

if submit:

    prediction = model.predict(
        input_data
    )

    probability = model.predict_proba(
        input_data
    )

    cancel_probability = round(
        probability[0][1] * 100,
        2
    )

    st.write(
        f"Cancellation Probability: {cancel_probability}%"
    )

    if prediction[0] == 1:

        st.error(
            "Booking Likely To Be Cancelled"
        )

    else:

        st.success(
            "Booking Likely To Be Confirmed"
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

st.subheader(
    "Classification Report"
)

report = classification_report(
    y_test,
    y_pred
)

st.text(report)

