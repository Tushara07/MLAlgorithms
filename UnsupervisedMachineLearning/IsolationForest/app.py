import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "fraud_detection_results_sample.csv"
    )
)
features = df.drop(
    columns=["Class"],
    errors="ignore"
)

scaled_data = scaler.transform(
    features
)

predictions = model.predict(
    scaled_data
)

df["Predicted_Class"] = np.where(
    predictions == -1,
    1,
    0
)

df["Transaction_Type"] = df[
    "Predicted_Class"
].map(
    {
        0: "Normal Transaction",
        1: "Potential Fraud"
    }
)

model = pickle.load(
    open(
        os.path.join(
            BASE_DIR,
            "isolation_forest_model.pkl"
        ),
        "rb"
    )
)

scaler = pickle.load(
    open(
        os.path.join(
            BASE_DIR,
            "scaler.pkl"
        ),
        "rb"
    )
)

st.set_page_config(
    page_title="Credit Card Fraud Detection using Isolation Forest",
    layout="wide"
)

st.sidebar.title("Dashboard")

show_dataset = st.sidebar.checkbox(
    "Dataset Details"
)

show_analysis = st.sidebar.checkbox(
    "Fraud Analysis"
)

show_visuals = st.sidebar.checkbox(
    "Visualizations"
)

st.title(
    "Credit Card Fraud Detection using Isolation Forest"
)

st.write(
    "This project uses Isolation Forest to identify potentially fraudulent credit card transactions."
)

st.header(
    "Upload Test Dataset"
)

uploaded_file = st.file_uploader(
    "Upload a CSV file containing transaction records",
    type=["csv"]
)

if uploaded_file is not None:

    test_df = pd.read_csv(
        uploaded_file
    )

    st.subheader(
        "Uploaded Dataset"
    )

    st.dataframe(
        test_df.head(),
        use_container_width=True
    )

    required_features = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount"
    ]

    missing_cols = [
        col
        for col in required_features
        if col not in test_df.columns
    ]

    if len(missing_cols) > 0:

        st.error(
            f"Missing Columns: {missing_cols}"
        )

    else:

        scaled_data = scaler.transform(
            test_df[
                required_features
            ]
        )

        predictions = model.predict(
            scaled_data
        )

        anomaly_scores = (
            model.decision_function(
                scaled_data
            )
        )

        predictions = np.where(
            predictions == -1,
            1,
            0
        )

        fraud_names = {
            0: "Normal Transaction",
            1: "Potential Fraud"
        }

        results_df = test_df.copy()

        results_df[
            "Prediction"
        ] = predictions

        results_df[
            "Transaction_Type"
        ] = results_df[
            "Prediction"
        ].map(
            fraud_names
        )

        results_df[
            "Anomaly_Score"
        ] = anomaly_scores

        st.subheader(
            "Analysis Results"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Records",
                len(results_df)
            )

        with col2:

            st.metric(
                "Normal Transactions",
                int(
                    (
                        results_df[
                            "Prediction"
                        ] == 0
                    ).sum()
                )
            )

        with col3:

            st.metric(
                "Potential Frauds",
                int(
                    (
                        results_df[
                            "Prediction"
                        ] == 1
                    ).sum()
                )
            )

        st.dataframe(
            results_df,
            use_container_width=True
        )

        csv = results_df.to_csv(
            index=False
        )

        st.download_button(
            label="Download Analysis Results",
            data=csv,
            file_name="fraud_analysis_results.csv",
            mime="text/csv"
        )


st.header("Train Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Transactions",
        len(df)
    )

with col2:
    st.metric(
        "Actual Frauds",
        int(
            (df["Class"] == 1).sum()
        )
    )

with col3:
    st.metric(
        "Detected Frauds",
        int(
            (
                (df["Class"] == 1)
                &
                (df["Predicted_Class"] == 1)
            ).sum()
        )
    )

with col4:
    st.metric(
        "Potential Frauds Found",
        int(
            (
                df["Predicted_Class"] == 1
            ).sum()
        )
    )

if show_dataset:

    st.header(
        "Dataset Details"
    )

    with st.container(border=True):

        st.subheader(
            "Dataset Shape"
        )

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

if show_analysis:

    st.header(
        "Fraud Analysis"
    )

    with st.container(border=True):

        st.subheader(
            "Transaction Type Distribution"
        )

        distribution = (
            df["Transaction_Type"]
            .value_counts()
        )

        st.dataframe(
            distribution,
            use_container_width=True
        )

    with st.container(border=True):

        st.subheader(
            "Actual vs Predicted Fraud"
        )

        comparison = pd.crosstab(
            df["Class"],
            df["Predicted_Class"]
        )

        st.dataframe(
            comparison,
            use_container_width=True
        )

if show_visuals:

    st.header(
        "Visualizations"
    )

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Predicted Fraud Distribution"
            )

            fig1, ax1 = plt.subplots(
                figsize=(4, 3)
            )

            df[
                "Predicted_Class"
            ].value_counts().sort_index().plot(
                kind="bar",
                ax=ax1
            )

            ax1.set_xlabel(
                "Class"
            )

            ax1.set_ylabel(
                "Count"
            )

            plt.tight_layout()

            st.pyplot(
                fig1
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Transaction Amount Distribution"
            )

            fig2, ax2 = plt.subplots(
                figsize=(4, 3)
            )

            ax2.hist(
                df["Amount"],
                bins=50
            )

            ax2.set_xlabel(
                "Amount"
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
            "Fraud vs Normal Transactions"
        )

        fig3, ax3 = plt.subplots(
            figsize=(5, 3)
        )

        df["Transaction_Type"].value_counts().plot(
            kind="bar",
            ax=ax3
        )

        ax3.set_xlabel(
            "Transaction Type"
        )

        ax3.set_ylabel(
            "Count"
        )

        plt.tight_layout()

        st.pyplot(
            fig3
        )