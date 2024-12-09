import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from io import BytesIO

# Set the seaborn style for better aesthetics
sns.set_style("whitegrid")

def upload_dataset():
    """
    Allows users to upload a dataset in CSV, Excel, or other supported formats.

    Returns:
        pd.DataFrame: The uploaded dataset as a pandas DataFrame.
    """
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("Dataset uploaded successfully!")
            st.write(f"Dataset Shape: {df.shape}")
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    return None

def filter_cases(df):
    """
    Provides functionality to filter instances (rows) based on user-defined criteria.

    Args:
        df (pd.DataFrame): The original dataset.

    Returns:
        pd.DataFrame: The filtered dataset.
    """
    st.sidebar.header("Filter Cases")
    filtered_df = df.copy()
    for column in df.select_dtypes(include=['number', 'object', 'category']).columns:
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            options = df[column].dropna().unique().tolist()
            selected = st.sidebar.multiselect(f"Select {column}", options, default=options)
            if selected:
                filtered_df = filtered_df[filtered_df[column].isin(selected)]
        else:
            min_val = float(df[column].min())
            max_val = float(df[column].max())
            step = (max_val - min_val) / 100 if (max_val - min_val) != 0 else 1
            selected_range = st.sidebar.slider(
                f"Select range for {column}",
                min_val, max_val, (min_val, max_val), step=step
            )
            filtered_df = filtered_df[
                (filtered_df[column] >= selected_range[0]) & 
                (filtered_df[column] <= selected_range[1])
            ]
    st.write(f"Number of records after filtering: {filtered_df.shape[0]}")
    return filtered_df

def aggregate_scores(df):
    """
    Enables users to create a composite score by aggregating selected numerical columns.

    Args:
        df (pd.DataFrame): The dataset to aggregate.

    Returns:
        pd.DataFrame: Dataset with an additional 'Composite_Score' column.
    """
    st.sidebar.header("Score Aggregation")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    selected_columns = st.sidebar.multiselect(
        "Select columns to aggregate into a score", 
        numeric_columns,
        default=numeric_columns[:2]  # Default selection
    )
    
    if selected_columns:
        aggregation_method = st.sidebar.selectbox(
            "Select aggregation method", 
            ["Sum", "Mean", "Weighted Sum"]
        )
        if aggregation_method == "Sum":
            df['Composite_Score'] = df[selected_columns].sum(axis=1)
        elif aggregation_method == "Mean":
            df['Composite_Score'] = df[selected_columns].mean(axis=1)
        elif aggregation_method == "Weighted Sum":
            weights = {}
            st.sidebar.write("### Assign Weights")
            for col in selected_columns:
                weights[col] = st.sidebar.number_input(
                    f"Weight for {col}", 
                    value=1.0, 
                    step=0.1
                )
            weights_series = pd.Series(weights)
            df['Composite_Score'] = df[selected_columns].mul(weights_series).sum(axis=1)
        st.success("Composite score created successfully!")
    else:
        st.warning("Please select at least one column to aggregate.")
    return df

def plot_monovariate_distribution(df):
    """
    Visualizes the distribution of individual numerical variables.

    Args:
        df (pd.DataFrame): The dataset to visualize.
    """
    st.header("Monovariate Distribution")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("No numerical columns available for visualization.")
        return
    selected_column = st.selectbox("Select a numerical column to visualize", numeric_columns)
    
    if selected_column:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_column].dropna(), kde=True, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {selected_column}", fontsize=16)
        ax.set_xlabel(selected_column, fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        st.pyplot(fig)

def plot_multivariate_distribution(df):
    """
    Visualizes the distribution of multiple numerical variables using pair plots or scatter plots.

    Args:
        df (pd.DataFrame): The dataset to visualize.
    """
    st.header("Multivariate Distribution")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_columns) < 2:
        st.warning("At least two numerical columns are required for multivariate visualization.")
        return
    selected_columns = st.multiselect(
        "Select numerical columns to visualize", 
        numeric_columns, 
        default=numeric_columns[:2]
    )
    
    if len(selected_columns) >= 2:
        plot_type = st.selectbox("Select plot type", ["Pair Plot", "Scatter Plot"])
        if plot_type == "Pair Plot":
            fig = sns.pairplot(df[selected_columns].dropna())
            st.pyplot(fig)
        elif plot_type == "Scatter Plot":
            x_axis = st.selectbox("X-axis", selected_columns, index=0)
            y_axis = st.selectbox("Y-axis", selected_columns, index=1)
            if x_axis and y_axis:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax, hue='Outlier' if 'Outlier' in df.columns else None, palette={True: 'red', False: 'blue'})
                ax.set_title(f"Scatter Plot of {x_axis} vs {y_axis}", fontsize=16)
                st.pyplot(fig)
    else:
        st.warning("Please select at least two numerical columns for multivariate visualization.")

def choose_statistical_test():
    """
    Allows users to select a statistical test for outlier detection.

    Returns:
        str: The name of the selected statistical test.
    """
    st.sidebar.header("Outlier Detection")
    test = st.sidebar.selectbox(
        "Select a statistical test to detect outliers", 
        ["Z-Score", "IQR", "DBSCAN", "Isolation Forest"]
    )
    return test

def identify_outliers(df, test):
    """
    Identifies outliers in the dataset based on the selected statistical test.

    Args:
        df (pd.DataFrame): The dataset to analyze.
        test (str): The statistical test selected for outlier detection.

    Returns:
        pd.DataFrame: Dataset with an additional 'Outlier' column indicating outliers.
    """
    st.header("Outlier Identification")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("No numerical columns available for outlier detection.")
        return df
    selected_columns = st.multiselect(
        "Select numerical columns for outlier detection", 
        numeric_columns, 
        default=numeric_columns[:2]
    )
    
    if selected_columns:
        if test == "Z-Score":
            threshold = st.sidebar.number_input("Z-Score Threshold", value=3.0, step=0.1)
            z_scores = np.abs(stats.zscore(df[selected_columns].dropna()))
            outliers = (z_scores > threshold).any(axis=1)
        elif test == "IQR":
            multiplier = st.sidebar.number_input("IQR Multiplier", value=1.5, step=0.1)
            Q1 = df[selected_columns].quantile(0.25)
            Q3 = df[selected_columns].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            outliers = ((df[selected_columns] < lower_bound) | (df[selected_columns] > upper_bound)).any(axis=1)
        elif test == "DBSCAN":
            eps = st.sidebar.number_input("DBSCAN eps", value=0.5, step=0.1)
            min_samples = st.sidebar.number_input("DBSCAN min_samples", value=5, step=1)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[selected_columns].dropna())
            db = DBSCAN(eps=eps, min_samples=min_samples)
            db.fit(scaled_data)
            labels = db.labels_
            outliers = labels == -1
            # Initialize a Series with all False
            outliers_series = pd.Series(False, index=df.index)
            outliers_series.loc[df[selected_columns].dropna().index] = outliers
            outliers = outliers_series
        elif test == "Isolation Forest":
            contamination = st.sidebar.number_input(
                "Isolation Forest Contamination", 
                value=0.05, 
                min_value=0.0, 
                max_value=0.5, 
                step=0.01
            )
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            iso_forest.fit(df[selected_columns].dropna())
            preds = iso_forest.predict(df[selected_columns].dropna())
            outliers = preds == -1
            # Initialize a Series with all False
            outliers_series = pd.Series(False, index=df.index)
            outliers_series.loc[df[selected_columns].dropna().index] = outliers
            outliers = outliers_series

        # Add the 'Outlier' column
        df['Outlier'] = outliers
        num_outliers = df['Outlier'].sum()
        st.write(f"Number of outliers detected: {num_outliers}")
        
        # Optionally, display some statistics or a table of outliers
        if num_outliers > 0:
            st.subheader("Outlier Records")
            st.write(df[df['Outlier']])
    else:
        st.warning("Please select at least one numerical column for outlier detection.")
    return df

def download_enhanced_dataset(df):
    """
    Provides an option for users to download the enhanced dataset with outlier flags.

    Args:
        df (pd.DataFrame): The enhanced dataset.
    """
    st.header("Download Enhanced Dataset")
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.save()
        processed_data = output.getvalue()
        return processed_data

    excel_data = to_excel(df)
    st.download_button(
        label="Download dataset with Outlier Flags",
        data=excel_data,
        file_name='enhanced_dataset.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

def main():
    """
    The main function that orchestrates the Streamlit application.
    """
    st.set_page_config(
        page_title="Outlier Detection and Analysis App",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📊 Outlier Detection and Analysis App")
    st.markdown("""
    This application allows you to upload a dataset, filter the data based on your criteria, 
    create composite scores, visualize distributions, perform outlier detection using various statistical tests, 
    and download the enhanced dataset with outlier flags.
    """)
    
    # Step 1: Upload Dataset
    df = upload_dataset()
    if df is not None:
        # Step 2: Filter Cases
        filtered_df = filter_cases(df)
        
        # Step 3: Aggregate Scores
        aggregated_df = aggregate_scores(filtered_df)
        
        # Step 4: Visualizations
        col1, col2 = st.columns(2)
        with col1:
            plot_monovariate_distribution(aggregated_df)
        with col2:
            plot_multivariate_distribution(aggregated_df)
        
        # Step 5: Choose Statistical Test
        selected_test = choose_statistical_test()
        
        # Step 6: Identify Outliers
        outlier_df = identify_outliers(aggregated_df, selected_test)
        
        # Step 7: Download Enhanced Dataset
        download_enhanced_dataset(outlier_df)
        
        # Optional: Display the enhanced dataset
        st.header("Enhanced Dataset")
        st.dataframe(outlier_df)
        
        # Optionally, provide a summary or statistics
        st.subheader("Dataset Statistics")
        st.write(outlier_df.describe())

if __name__ == "__main__":
    main()
