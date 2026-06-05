import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(BASE_DIR, "clustered_customers.csv")
)

scaler = pickle.load(
    open(
        os.path.join(BASE_DIR, "scaler.pkl"),
        "rb"
    )
)

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

cluster_names = {
    -1: "Outlier Customers",
    0: "Young High Spenders",
    1: "Low Income Customers",
    2: "Senior Regular Customers",
    3: "Young Regular Customers",
    4: "Premium Customers",
    5: "High Income Low Spenders"
}

cluster_centers = (
    df.groupby("Cluster")[features]
    .mean()
)

st.set_page_config(
    page_title="Customer Segmentation using DBSCAN",
    layout="wide"
)

st.sidebar.title("Dashboard")

show_dataset = st.sidebar.checkbox("Dataset Details")
show_cluster = st.sidebar.checkbox("Cluster Analysis")
show_visuals = st.sidebar.checkbox("Visualizations")

st.title("Customer Purchase Behavior Clustering using DBSCAN")

st.write(
    "Enter customer details to identify the customer purchase behaviour."
)

age = st.number_input(
    "Age",
    min_value=1,
    value=30
)

income = st.number_input(
    "Annual Income (k$)",
    min_value=0,
    value=50
)

spending_score = st.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)

if st.button("Predict Cluster"):

    input_data = pd.DataFrame(
        [[
            age,
            income,
            spending_score
        ]],
        columns=features
    )

    scaled_input = scaler.transform(
        input_data
    )

    scaled_centers = scaler.transform(
        cluster_centers
    )

    distances = np.linalg.norm(
        scaled_centers - scaled_input,
        axis=1
    )

    nearest_index = np.argmin(
        distances
    )

    cluster = cluster_centers.index[
        nearest_index
    ]

    st.success(
        f"Customer belongs to Cluster {cluster}"
    )

    st.info(
        cluster_names.get(
            cluster,
            "Unknown Cluster"
        )
    )

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

    summary = (
        df.groupby("Cluster")[
            features
        ]
        .mean()
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        with st.container(border=True):

            st.subheader("Cluster Summary")

            st.dataframe(
                summary,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Cluster Distribution"
            )

            cluster_counts = (
                df["Cluster"]
                .value_counts()
                .sort_index()
            )

            fig, ax = plt.subplots(
                figsize=(4, 2.5)
            )

            cluster_counts.plot(
                kind="bar",
                ax=ax
            )

            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")

            plt.tight_layout()

            st.pyplot(fig)

    noise_count = (
        df["Cluster"] == -1
    ).sum()

    st.metric(
        "Outlier Customers",
        noise_count
    )

if show_visuals:

    st.header("Visualizations")

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Income Distribution"
            )

            fig1, ax1 = plt.subplots(
                figsize=(4, 2.5)
            )

            ax1.hist(
                df["Annual Income (k$)"],
                bins=20
            )

            ax1.set_xlabel(
                "Annual Income (k$)"
            )

            ax1.set_ylabel(
                "Frequency"
            )

            plt.tight_layout()

            st.pyplot(fig1)

    with col2:

        with st.container(border=True):

            st.subheader(
                "Spending Score Distribution"
            )

            fig2, ax2 = plt.subplots(
                figsize=(4, 2.5)
            )

            ax2.hist(
                df["Spending Score (1-100)"],
                bins=20
            )

            ax2.set_xlabel(
                "Spending Score"
            )

            ax2.set_ylabel(
                "Frequency"
            )

            plt.tight_layout()

            st.pyplot(fig2)

    with st.container(border=True):

        st.subheader(
            "PCA Cluster Visualization"
        )

        scaled_features = scaler.transform(
            df[features]
        )

        pca = PCA(
            n_components=2
        )

        pca_features = pca.fit_transform(
            scaled_features
        )

        fig3, ax3 = plt.subplots(
            figsize=(5, 3)
        )

        ax3.scatter(
            pca_features[:, 0],
            pca_features[:, 1],
            c=df["Cluster"],
            s=20,
            alpha=0.7
        )

        ax3.set_xlabel("PCA 1")
        ax3.set_ylabel("PCA 2")

        plt.tight_layout()

        st.pyplot(fig3)