import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(BASE_DIR, "clustered_players.csv")
)

gmm = pickle.load(
    open(
        os.path.join(BASE_DIR, "gmm_model.pkl"),
        "rb"
    )
)

scaler = pickle.load(
    open(
        os.path.join(BASE_DIR, "scaler.pkl"),
        "rb"
    )
)

features = [
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic"
]

cluster_names = {
    0: "Attacking Midfielders",
    1: "Defenders",
    2: "All-Round Players",
    3: "Attackers"
}

st.set_page_config(
    page_title="Football Player Classification using GMM",
    layout="wide"
)

st.sidebar.title("Dashboard")

show_dataset = st.sidebar.checkbox("Dataset Details")
show_cluster = st.sidebar.checkbox("Cluster Analysis")
show_visuals = st.sidebar.checkbox("Visualizations")

st.title(
    "Football Player Classification using Gaussian Mixture Model"
)

st.write(
    "Enter player attributes to identify the player category."
)

pace = st.number_input(
    "Pace",
    min_value=1,
    max_value=100,
    value=70
)

shooting = st.number_input(
    "Shooting",
    min_value=1,
    max_value=100,
    value=60
)

passing = st.number_input(
    "Passing",
    min_value=1,
    max_value=100,
    value=60
)

dribbling = st.number_input(
    "Dribbling",
    min_value=1,
    max_value=100,
    value=60
)

defending = st.number_input(
    "Defending",
    min_value=1,
    max_value=100,
    value=50
)

physic = st.number_input(
    "Physical",
    min_value=1,
    max_value=100,
    value=60
)

if st.button("Predict Cluster"):

    input_data = pd.DataFrame(
        [[
            pace,
            shooting,
            passing,
            dribbling,
            defending,
            physic
        ]],
        columns=features
    )

    scaled_input = scaler.transform(
        input_data
    )

    cluster = gmm.predict(
        scaled_input
    )[0]

    probabilities = gmm.predict_proba(
        scaled_input
    )[0]

    st.success(
        f"Player belongs to Cluster {cluster}"
    )

    st.info(
        cluster_names.get(
            cluster,
            "Unknown Cluster"
        )
    )

    st.subheader(
        "Cluster Probabilities"
    )

    probability_df = pd.DataFrame(
        {
            "Cluster": [
                cluster_names[i]
                for i in range(
                    len(probabilities)
                )
            ],
            "Probability (%)": (
                probabilities * 100
            ).round(2)
        }
    )

    st.dataframe(
        probability_df,
        use_container_width=True
    )

if show_dataset:

    st.header("Dataset Details")

    with st.container(border=True):

        st.subheader("Dataset Shape")

        st.write(
            df.shape
        )

        st.subheader(
            "Complete Dataset"
        )

        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

        st.subheader(
            "Statistical Summary"
        )

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

    summary.index = [
        cluster_names[i]
        for i in summary.index
    ]

    col1, col2 = st.columns(
        [2, 1]
    )

    with col1:

        with st.container(border=True):

            st.subheader(
                "Cluster Summary"
            )

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

            ax.set_xlabel(
                "Cluster"
            )

            ax.set_ylabel(
                "Count"
            )

            plt.tight_layout()

            st.pyplot(
                fig
            )

if show_visuals:

    st.header(
        "Visualizations"
    )

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Pace Distribution"
            )

            fig1, ax1 = plt.subplots(
                figsize=(4, 2.5)
            )

            ax1.hist(
                df["pace"],
                bins=20
            )

            ax1.set_xlabel(
                "Pace"
            )

            ax1.set_ylabel(
                "Frequency"
            )

            plt.tight_layout()

            st.pyplot(
                fig1
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Shooting Distribution"
            )

            fig2, ax2 = plt.subplots(
                figsize=(4, 2.5)
            )

            ax2.hist(
                df["shooting"],
                bins=20
            )

            ax2.set_xlabel(
                "Shooting"
            )

            ax2.set_ylabel(
                "Frequency"
            )

            plt.tight_layout()

            st.pyplot(
                fig2
            )

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
            s=10,
            alpha=0.7
        )

        ax3.set_xlabel(
            "PCA 1"
        )

        ax3.set_ylabel(
            "PCA 2"
        )

        plt.tight_layout()

        st.pyplot(
            fig3
        )