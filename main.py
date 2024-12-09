import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Function to handle the upload of the dataset file
def upload_dataset():
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            dataset = pd.read_csv(uploaded_file)
        else:
            dataset = pd.read_excel(uploaded_file)
        return dataset
    return None

# Function to filter instances based on user-defined criteria
def filter_cases(dataset, criteria):
    filtered_dataset = dataset.query(criteria)
    return filtered_dataset

# Function to create scores by aggregating specified columns
def aggregate_scores(dataset, columns):
    dataset['score'] = dataset[columns].sum(axis=1)
    return dataset

# Function to visualize the distribution of a single variable
def visualize_monovariate(dataset, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(dataset[column], kde=True)
    st.pyplot(plt)

# Function to visualize the distribution of multiple variables
def visualize_multivariate(dataset, columns):
    plt.figure(figsize=(10, 6))
    sns.pairplot(dataset[columns])
    st.pyplot(plt)

# Function to allow users to choose and apply a statistical test for outlier detection
def choose_statistical_test(dataset, test_type):
    if test_type == 'Z-Score':
        z_scores = np.abs(stats.zscore(dataset.select_dtypes(include=[np.number])))
        outliers = (z_scores > 3).any(axis=1)
    elif test_type == 'IQR':
        Q1 = dataset.select_dtypes(include=[np.number]).quantile(0.25)
        Q3 = dataset.select_dtypes(include=[np.number]).quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((dataset.select_dtypes(include=[np.number]) < (Q1 - 1.5 * IQR)) | (dataset.select_dtypes(include=[np.number]) > (Q3 + 1.5 * IQR))).any(axis=1)
    else:
        st.error("Unsupported test type")
        return None
    return outliers

# Function to identify outliers based on the results of the statistical test
def identify_outliers(dataset, outliers):
    dataset['outlier'] = outliers
    return dataset

# Function to provide an option for users to download the dataset enhanced with a filter variable to select/deselect outliers
def download_enhanced_dataset(dataset):
    st.download_button(
        label="Download enhanced dataset as CSV",
        data=dataset.to_csv(index=False),
        file_name='enhanced_dataset.csv',
        mime='text/csv',
    )

# Streamlit app
def main():
    st.title("Outlier Detection App")

    # Upload dataset
    dataset = upload_dataset()
    if dataset is not None:
        st.write("Dataset uploaded successfully")
        st.dataframe(dataset.head())

        # Filter cases
        st.sidebar.header("Filter Cases")
        criteria = st.sidebar.text_input("Enter filter criteria (e.g., 'column_name > value')")
        if criteria:
            dataset = filter_cases(dataset, criteria)
            st.write("Filtered dataset")
            st.dataframe(dataset.head())

        # Aggregate scores
        st.sidebar.header("Aggregate Scores")
        columns_to_aggregate = st.sidebar.multiselect("Select columns to aggregate", dataset.columns)
        if columns_to_aggregate:
            dataset = aggregate_scores(dataset, columns_to_aggregate)
            st.write("Dataset with aggregated scores")
            st.dataframe(dataset.head())

        # Visualize monovariate distribution
        st.sidebar.header("Monovariate Distribution")
        column_to_visualize = st.sidebar.selectbox("Select column to visualize", dataset.columns)
        if column_to_visualize:
            visualize_monovariate(dataset, column_to_visualize)

        # Visualize multivariate distribution
        st.sidebar.header("Multivariate Distribution")
        columns_to_visualize = st.sidebar.multiselect("Select columns to visualize", dataset.columns)
        if columns_to_visualize:
            visualize_multivariate(dataset, columns_to_visualize)

        # Choose statistical test
        st.sidebar.header("Statistical Test for Outlier Detection")
        test_type = st.sidebar.selectbox("Select test type", ['Z-Score', 'IQR'])
        if test_type:
            outliers = choose_statistical_test(dataset, test_type)
            if outliers is not None:
                dataset = identify_outliers(dataset, outliers)
                st.write("Dataset with identified outliers")
                st.dataframe(dataset.head())

        # Download enhanced dataset
        st.sidebar.header("Download Enhanced Dataset")
        download_enhanced_dataset(dataset)

if __name__ == "__main__":
    main()
