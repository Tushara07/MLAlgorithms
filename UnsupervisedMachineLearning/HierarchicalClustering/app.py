import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(BASE_DIR, "clustered_pokemon.csv")
)

scaler = pickle.load(
    open(
        os.path.join(BASE_DIR, "scaler.pkl"),
        "rb"
    )
)

features = [
    "HP",
    "Attack",
    "Defense",
    "Sp. Atk",
    "Sp. Def",
    "Speed"
]

cluster_names = {
    0: "Balanced Pokemon",
    1: "Defensive Pokemon",
    2: "Fast Offensive Pokemon",
    3: "Elite Powerful Pokemon"
}

cluster_centers = (
    df.groupby("Cluster")[features]
    .mean()
)

st.set_page_config(
    page_title="Pokemon Hierarchical Clustering",
    layout="wide"
)

st.sidebar.title("Dashboard")

show_dataset = st.sidebar.checkbox("Dataset Details")
show_cluster = st.sidebar.checkbox("Cluster Analysis")
show_visuals = st.sidebar.checkbox("Visualizations")

st.title("Pokemon Hierarchical Clustering")

st.write(
    "Enter Pokemon statistics to identify its cluster."
)

col1, col2 = st.columns(2)

with col1:

    hp = st.number_input(
        "Hit Points (HP)",
        min_value=1,
        value=50
    )

    attack = st.number_input(
        "Attack",
        min_value=1,
        value=50
    )

    defense = st.number_input(
        "Defense",
        min_value=1,
        value=50
    )

with col2:

    sp_atk = st.number_input(
        "Special Attack",
        min_value=1,
        value=50
    )

    sp_def = st.number_input(
        "Special Defense",
        min_value=1,
        value=50
    )

    speed = st.number_input(
        "Speed",
        min_value=1,
        value=50
    )

if st.button("Predict Cluster"):

    input_data = pd.DataFrame(
        [[
            hp,
            attack,
            defense,
            sp_atk,
            sp_def,
            speed
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

    cluster = np.argmin(
        distances
    )

    st.success(
        f"Pokemon belongs to Cluster {cluster}"
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

    summary = df.groupby("Cluster")[
        features
    ].mean()

    summary.columns = [
        "Hit Points",
        "Attack",
        "Defense",
        "Special Attack",
        "Special Defense",
        "Speed"
    ]

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

            st.subheader("Cluster Distribution")

            cluster_counts = (
                df["Cluster"]
                .value_counts()
                .sort_index()
            )

            fig, ax = plt.subplots(
                figsize=(4, 2.1)
            )

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

            st.subheader("Attack Distribution")

            fig1, ax1 = plt.subplots(
                figsize=(4, 2.5)
            )

            ax1.hist(
                df["Attack"],
                bins=20
            )

            ax1.set_xlabel("Attack")
            ax1.set_ylabel("Frequency")

            plt.tight_layout()

            st.pyplot(fig1)

    with col2:

        with st.container(border=True):

            st.subheader("Speed Distribution")

            fig2, ax2 = plt.subplots(
                figsize=(4, 2.5)
            )

            ax2.hist(
                df["Speed"],
                bins=20
            )

            ax2.set_xlabel("Speed")
            ax2.set_ylabel("Frequency")

            plt.tight_layout()

            st.pyplot(fig2)

    with st.container(border=True):

        st.subheader("Dendrogram")

        sample_size = min(
            100,
            len(df)
        )

        sample_data = df[
            features
        ].sample(
            sample_size,
            random_state=42
        )

        sample_scaled = scaler.transform(
            sample_data
        )

        fig3, ax3 = plt.subplots(
            figsize=(8, 4)
        )

        dendrogram(
            linkage(
                sample_scaled,
                method="ward"
            ),
            ax=ax3
        )

        plt.tight_layout()

        st.pyplot(fig3)

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

        fig4, ax4 = plt.subplots(
            figsize=(5, 3)
        )

        ax4.scatter(
            pca_features[:, 0],
            pca_features[:, 1],
            c=df["Cluster"],
            s=10,
            alpha=0.7
        )

        ax4.set_xlabel("PCA 1")
        ax4.set_ylabel("PCA 2")

        plt.tight_layout()

        st.pyplot(fig4)