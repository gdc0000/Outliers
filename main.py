import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import base64
from sklearn.cluster import DBSCAN

# Function to upload dataset
def upload_dataset():
    """
    Allow users to upload a dataset.
    Returns:
        pandas.DataFrame: The uploaded dataset.
    """
    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    return None

# Function to filter cases
def filter_cases(df):
    """
    Filter instances (cases) based on user-defined criteria.
    Args:
        df (pandas.DataFrame): The dataset to filter.
    Returns:
        pandas.DataFrame: The filtered dataset.
    """
    st.sidebar.header("Filter Cases")
    filter_column = st.sidebar.selectbox("Select a column to filter", df.columns)
    filter_value = st.sidebar.text_input(f"Enter value to filter {filter_column}")
    if filter_value:
        try:
            df = df[df[filter_column] == filter_value]
        except:
            st.error("Invalid filter value")
    return df

# Function to aggregate scores
def aggregate_scores(df):
    """
    Enable users to create scores by aggregating columns (variables).
    Args:
        df (pandas.DataFrame): The dataset to aggregate.
    Returns:
        pandas.DataFrame: The dataset with aggregated scores.
    """
    st.sidebar.header("Aggregate Scores")
    selected_columns = st.sidebar.multiselect("Select columns to aggregate", df.columns)
    if selected_columns:
        score_name = st.sidebar.text_input("Enter a name for the aggregated score")
        if score_name:
            df[score_name] = df[selected_columns].mean(axis=1)  # Example aggregation: mean
    return df

# Function to visualize monovariate distribution
def visualize_monovariate_distribution(df):
    """
    Visualize the distribution of individual variables.
    Args:
        df (pandas.DataFrame): The dataset to visualize.
    """
    st.header("Monovariate Distribution")
    column = st.selectbox("Select a column to visualize", df.columns)
    if column:
        fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")
        st.plotly_chart(fig)

# Function to visualize multivariate distribution
def visualize_multivariate_distribution(df):
    """
    Visualize the distribution of multiple variables.
    Args:
        df (pandas.DataFrame): The dataset to visualize.
    """
    st.header("Multivariate Distribution")
    selected_columns = st.multiselect("Select columns to visualize", df.columns)
    if selected_columns:
        fig = px.scatter_matrix(df, dimensions=selected_columns, title="Scatter Matrix")
        st.plotly_chart(fig)

# Function to select statistical test
def select_statistical_test():
    """
    Allow users to choose the best statistical test to detect outliers.
    Returns:
        str: The selected statistical test.
    """
    st.sidebar.header("Statistical Tests")
    test = st.sidebar.selectbox("Select a statistical test", ["Z-Score", "IQR", "DBSCAN"])
    return test

# Function to identify outliers
def identify_outliers(df, test):
    """
    Identify outliers based on the chosen statistical test.
    Args:
        df (pandas.DataFrame): The dataset to analyze.
        test (str): The selected statistical test.
    Returns:
        pandas.DataFrame: The dataset with an 'outlier' column.
    """
    if test == "Z-Score":
        z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
        df['outlier'] = (z_scores > 3).any(axis=1)
    elif test == "IQR":
        q1 = df.select_dtypes(include=[np.number]).quantile(0.25)
        q3 = df.select_dtypes(include=[np.number]).quantile(0.75)
        iqr = q3 - q1
        df['outlier'] = ((df.select_dtypes(include=[np.number]) < (q1 - 1.5 * iqr)) | (df.select_dtypes(include=[np.number]) > (q3 + 1.5 * iqr))).any(axis=1)
    elif test == "DBSCAN":
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        df['outlier'] = dbscan.fit_predict(df.select_dtypes(include=[np.number])) == -1
    return df

# Function to download enhanced dataset
def download_enhanced_dataset(df):
    """
    Provide an option for users to download the dataset enhanced with a filter variable to select/deselect outliers.
    Args:
        df (pandas.DataFrame): The dataset to download.
    """
    st.header("Download Enhanced Dataset")
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="enhanced_dataset.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

# Main function
def main():
    st.title("Outlier Detection Application")
    df = upload_dataset()
    if df is not None:
        df = filter_cases(df)
        df = aggregate_scores(df)
        test = select_statistical_test()
        df = identify_outliers(df, test)
        visualize_monovariate_distribution(df)
        visualize_multivariate_distribution(df)
        download_enhanced_dataset(df)
