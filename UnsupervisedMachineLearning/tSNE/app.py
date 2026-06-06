import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

tsne_df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "tsne_gene_expression.csv"
    )
)

st.set_page_config(
    page_title="Leukemia Gene Expression Analysis using t-SNE",
    layout="wide"
)

st.title(
    "Leukemia Gene Expression Analysis using t-SNE"
)

st.write(
    "This project demonstrates dimensionality reduction and visualization of high-dimensional gene expression data using t-SNE."
)

st.header("t-SNE Overview")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Original Features",
        "22,283"
    )

with col2:

    st.metric(
        "Reduced Features",
        "2"
    )

with col3:

    st.metric(
        "Reduction",
        "99.99%"
    )

st.header(
    "Dataset Details"
)

with st.container(border=True):

    st.subheader(
        "Dataset Shape"
    )

    st.write(
        tsne_df.shape
    )

    st.subheader(
        "t-SNE Dataset"
    )

    st.dataframe(
        tsne_df,
        use_container_width=True,
        height=400
    )

    st.subheader(
        "Statistical Summary"
    )

    st.dataframe(
        tsne_df.describe(),
        use_container_width=True
    )

st.header(
    "t-SNE Statistics"
)

with st.container(border=True):

    stats_df = pd.DataFrame(
        {
            "Metric": [
                "Original Features",
                "Reduced Features",
                "Dimensionality Reduction"
            ],
            "Value": [
                22283,
                2,
                "99.99%"
            ]
        }
    )

    st.dataframe(
        stats_df,
        use_container_width=True
    )

st.header(
    "Cell Type Distribution"
)

with st.container(border=True):

    cell_counts = (
        tsne_df["Cell_Type"]
        .value_counts()
    )

    fig1, ax1 = plt.subplots(
        figsize=(6, 3)
    )

    cell_counts.plot(
        kind="bar",
        ax=ax1
    )

    ax1.set_xlabel(
        "Cell Type"
    )

    ax1.set_ylabel(
        "Count"
    )

    plt.tight_layout()

    st.pyplot(
        fig1
    )

st.header(
    "t-SNE Visualization"
)

with st.container(border=True):

    fig2, ax2 = plt.subplots(
        figsize=(10, 6)
    )

    cell_types = (
        tsne_df["Cell_Type"]
        .unique()
    )

    for cell_type in cell_types:

        subset = tsne_df[
            tsne_df["Cell_Type"]
            == cell_type
        ]

        ax2.scatter(
            subset["TSNE1"],
            subset["TSNE2"],
            s=40,
            alpha=0.7,
            label=cell_type
        )

    ax2.set_xlabel(
        "t-SNE Component 1"
    )

    ax2.set_ylabel(
        "t-SNE Component 2"
    )

    ax2.set_title(
        "Gene Expression Visualization using t-SNE"
    )

    ax2.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    st.pyplot(
        fig2
    )