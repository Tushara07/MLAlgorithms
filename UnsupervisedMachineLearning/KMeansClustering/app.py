import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

df = pd.read_csv("clustered_customers.csv")

st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

st.sidebar.title("Dashboard")

show_dataset = st.sidebar.checkbox("Dataset Details")
show_cluster = st.sidebar.checkbox("Cluster Analysis")
show_visuals = st.sidebar.checkbox("Visualizations")

st.title("Customer Segmentation using K-Means")

st.write("Enter customer details to predict the customer segment.")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Income",
        min_value=0,
        value=50000
    )

    age = st.number_input(
        "Age",
        min_value=1,
        value=30
    )

    total_spending = st.number_input(
        "Total Spending",
        min_value=0,
        value=500
    )

with col2:
    total_purchases = st.number_input(
        "Total Purchases",
        min_value=0,
        value=10
    )

    recency = st.number_input(
        "Recency",
        min_value=0,
        value=30
    )

if st.button("Predict Segment"):

    input_data = np.array([
        [
            income,
            age,
            total_spending,
            total_purchases,
            recency
        ]
    ])

    scaled_data = scaler.transform(input_data)

    cluster = model.predict(scaled_data)[0]

    st.success(f"Customer belongs to Cluster {cluster}")

    if cluster == 0:
        st.info("High-value customers")
    elif cluster == 1:
        st.info("Low-spending customers")
    elif cluster == 2:
        st.info("Frequent buyers")
    elif cluster == 3:
        st.info("Inactive or occasional customers")

if show_dataset:

    st.header("Dataset Details")

    with st.container(border=True):

        st.subheader("Dataset Shape")
        st.write(df.shape)

        st.subheader("Complete Dataset")
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

        st.subheader("Statistical Summary")
        st.dataframe(
            df.describe(),
            use_container_width=True
        )

if show_cluster:

    st.header("Cluster Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:

        with st.container(border=True):

            st.subheader("Cluster Summary")

            summary = df.groupby("Cluster")[
                [
                    "Income",
                    "Age",
                    "Total_Spending",
                    "Total_Purchases",
                    "Recency"
                ]
            ].mean()

            st.dataframe(
                summary,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader("Cluster Distribution")

            cluster_counts = df["Cluster"].value_counts().sort_index()

            fig, ax = plt.subplots(figsize=(4, 2.5))

            cluster_counts.plot(
                kind="bar",
                ax=ax
            )

            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")

            plt.tight_layout()

            st.pyplot(fig)

if show_visuals:

    st.header("Visualizations")

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader("Income Distribution")

            fig1, ax1 = plt.subplots(figsize=(4, 2.5))

            ax1.hist(df["Income"], bins=20)

            ax1.set_xlabel("Income")
            ax1.set_ylabel("Frequency")

            plt.tight_layout()

            st.pyplot(fig1)

    with col2:

        with st.container(border=True):

            st.subheader("Total Spending Distribution")

            fig2, ax2 = plt.subplots(figsize=(4, 2.5))

            ax2.hist(df["Total_Spending"], bins=20)

            ax2.set_xlabel("Total Spending")
            ax2.set_ylabel("Frequency")

            plt.tight_layout()

            st.pyplot(fig2)

    st.markdown("")

    col3, col4, col5 = st.columns([1, 2, 1])

    with col4:

        with st.container(border=True):

            st.subheader("PCA Cluster Visualization")

            features = df[
                [
                    "Income",
                    "Age",
                    "Total_Spending",
                    "Total_Purchases",
                    "Recency"
                ]
            ]

            scaled_features = scaler.transform(features)

            pca = PCA(n_components=2)

            pca_features = pca.fit_transform(
                scaled_features
            )

            fig3, ax3 = plt.subplots(figsize=(5, 3))

            ax3.scatter(
                pca_features[:, 0],
                pca_features[:, 1],
                c=df["Cluster"],
                s=10,
                alpha=0.7
            )

            ax3.set_xlabel("PCA 1")
            ax3.set_ylabel("PCA 2")

            plt.tight_layout()

            st.pyplot(fig3)