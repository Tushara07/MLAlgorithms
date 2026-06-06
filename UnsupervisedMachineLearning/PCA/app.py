import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pca_df = pd.read_csv(
    os.path.join(BASE_DIR, "pca_data.csv")
)

pca_model = pickle.load(
    open(
        os.path.join(BASE_DIR, "pca_model.pkl"),
        "rb"
    )
)

st.set_page_config(
    page_title="Human Activity Recognition using PCA",
    layout="wide"
)

st.title(
    "Human Activity Recognition using Principal Component Analysis"
)

st.write(
    "This project demonstrates dimensionality reduction using PCA on the Human Activity Recognition dataset."
)

st.header("PCA Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Original Features",
        561
    )

with col2:
    st.metric(
        "Reduced Features",
        2
    )

with col3:
    st.metric(
        "Variance Retained",
        "57.26%"
    )

with col4:
    st.metric(
        "95% Variance Components",
        102
    )

st.header("Dataset Details")

with st.container(border=True):

    st.subheader("Dataset Shape")

    st.write(
        pca_df.shape
    )

    st.subheader("PCA Dataset")

    st.dataframe(
        pca_df,
        use_container_width=True,
        height=400
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        pca_df.describe(),
        use_container_width=True
    )

st.header("PCA Statistics")

col1, col2 = st.columns(2)

with col1:

    with st.container(border=True):

        st.subheader(
            "Explained Variance Ratio"
        )

        variance_df = pd.DataFrame(
            {
                "Principal Component": [
                    "PC1",
                    "PC2"
                ],
                "Explained Variance (%)": [
                    50.69,
                    6.57
                ]
            }
        )

        st.dataframe(
            variance_df,
            use_container_width=True
        )

with col2:

    with st.container(border=True):

        st.subheader(
            "Dimensionality Reduction"
        )

        reduction_df = pd.DataFrame(
            {
                "Metric": [
                    "Original Features",
                    "Reduced Features",
                    "Components for 95% Variance",
                    "Reduction (%)"
                ],
                "Value": [
                    561,
                    2,
                    102,
                    81.82
                ]
            }
        )

        st.dataframe(
            reduction_df,
            use_container_width=True
        )

st.header("Visualizations")

col1, col2 = st.columns(2)

with col1:

    with st.container(border=True):

        st.subheader(
            "Explained Variance"
        )

        fig1, ax1 = plt.subplots(
            figsize=(5, 3)
        )

        ax1.bar(
            ["PC1", "PC2"],
            [50.69, 6.57]
        )

        ax1.set_ylabel(
            "Variance (%)"
        )

        ax1.set_title(
            "Explained Variance by Components"
        )

        plt.tight_layout()

        st.pyplot(
            fig1
        )

with col2:

    with st.container(border=True):

        st.subheader(
            "Variance Retention Summary"
        )

        summary_df = pd.DataFrame(
            {
                "Measure": [
                    "Variance Retained by PC1 & PC2",
                    "Components Needed for 95% Variance"
                ],
                "Value": [
                    "57.26%",
                    102
                ]
            }
        )

        st.dataframe(
            summary_df,
            use_container_width=True
        )

with st.container(border=True):

    st.subheader(
        "PCA Scatter Plot"
    )

    fig2, ax2 = plt.subplots(
        figsize=(8, 5)
    )

    activities = pca_df[
        "Activity"
    ].unique()

    for activity in activities:

        subset = pca_df[
            pca_df["Activity"] == activity
        ]

        ax2.scatter(
            subset["PC1"],
            subset["PC2"],
            s=10,
            alpha=0.7,
            label=activity
        )

    ax2.set_xlabel(
        "Principal Component 1"
    )

    ax2.set_ylabel(
        "Principal Component 2"
    )

    ax2.set_title(
        "Human Activity Distribution in PCA Space"
    )

    ax2.legend()

    plt.tight_layout()

    st.pyplot(
        fig2
    )   