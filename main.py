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
    uploaded_file = st.file_uploader("📥 Upload your dataset", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("✅ Dataset uploaded successfully!")
            st.write(f"**Dataset Shape:** {df.shape}")
            st.dataframe(df.head())
            return df
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
    return None

def filter_cases(df):
    """
    Provides functionality to filter instances (rows) based on user-defined conditions.

    Args:
        df (pd.DataFrame): The original dataset.

    Returns:
        pd.DataFrame: The filtered dataset.
    """
    st.header("🔍 Filter Cases")

    st.markdown("""
    ### Apply Filter Conditions
    Enter your filter conditions using pandas query syntax.
    - Use logical operators: `and`, `or`, `not`
    - Use comparison operators: `>`, `<`, `>=`, `<=`, `==`, `!=`
    - Enclose string values in quotes: `'value'` or `"value"`

    **Examples:**
    - `AGE > 30`
    - `INCOME < 50000 and AGE >= 25`
    - `CITY == 'New York' or CITY == 'Los Angeles'`
    """)

    filter_condition = st.text_area(
        "📋 Enter Filter Conditions",
        height=150,
        placeholder="e.g., AGE > 30 and INCOME < 50000"
    )

    if st.button("Apply Filter"):
        if filter_condition.strip() == "":
            st.warning("⚠️ Please enter at least one filter condition.")
            return df
        try:
            # Use query to filter the dataframe
            filtered_df = df.query(filter_condition, engine='python')
            st.success("✅ Filter applied successfully!")
            st.write(f"**Number of records after filtering:** {filtered_df.shape[0]}")
            st.dataframe(filtered_df.head())
            return filtered_df
        except Exception as e:
            st.error(f"❌ Error applying filter: {e}")
            st.info("🔍 Ensure your filter conditions are valid and match the dataset's column names and data types.")
            return df
    else:
        st.write(f"**Number of records after filtering:** {df.shape[0]}")
        return df

def aggregate_scores(df):
    """
    Enables users to create a composite score by aggregating selected numerical columns.

    Args:
        df (pd.DataFrame): The dataset to aggregate.

    Returns:
        pd.DataFrame: Dataset with an additional 'Composite_Score' column.
    """
    st.header("🧮 Score Aggregation")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("⚠️ No numerical columns available for aggregation.")
        return df

    selected_columns = st.multiselect(
        "📊 Select columns to aggregate into a score", 
        numeric_columns,
        default=numeric_columns[:2]  # Default selection
    )
    
    if selected_columns:
        aggregation_method = st.selectbox(
            "🔧 Select aggregation method", 
            ["Sum", "Mean", "Weighted Sum"]
        )
        if aggregation_method == "Sum":
            df['Composite_Score'] = df[selected_columns].sum(axis=1)
            st.success("✅ Composite score (Sum) created successfully!")
        elif aggregation_method == "Mean":
            df['Composite_Score'] = df[selected_columns].mean(axis=1)
            st.success("✅ Composite score (Mean) created successfully!")
        elif aggregation_method == "Weighted Sum":
            weights = {}
            st.markdown("### ⚖️ Assign Weights")
            for col in selected_columns:
                weights[col] = st.number_input(
                    f"Weight for `{col}`", 
                    value=1.0, 
                    step=0.1,
                    format="%.2f"
                )
            weights_series = pd.Series(weights)
            df['Composite_Score'] = df[selected_columns].mul(weights_series).sum(axis=1)
            st.success("✅ Composite score (Weighted Sum) created successfully!")
        st.write("### 📈 Composite Score Statistics")
        st.write(df[['Composite_Score']].describe())
    else:
        st.warning("⚠️ Please select at least one column to aggregate.")
    return df

def plot_monovariate_distribution(df):
    """
    Visualizes the distribution of individual numerical variables.

    Args:
        df (pd.DataFrame): The dataset to visualize.
    """
    st.header("📊 Monovariate Distribution")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("⚠️ No numerical columns available for visualization.")
        return
    selected_column = st.selectbox("Select a numerical column to visualize", numeric_columns)
    
    if selected_column:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_column].dropna(), kde=True, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of `{selected_column}`", fontsize=16)
        ax.set_xlabel(selected_column, fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        st.pyplot(fig)

def plot_multivariate_distribution(df):
    """
    Visualizes the distribution of multiple numerical variables using pair plots or scatter plots.

    Args:
        df (pd.DataFrame): The dataset to visualize.
    """
    st.header("📈 Multivariate Distribution")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_columns) < 2:
        st.warning("⚠️ At least two numerical columns are required for multivariate visualization.")
        return
    selected_columns = st.multiselect(
        "🔍 Select numerical columns to visualize", 
        numeric_columns, 
        default=numeric_columns[:2]
    )
    
    if len(selected_columns) >= 2:
        plot_type = st.selectbox("🔧 Select plot type", ["Pair Plot", "Scatter Plot"])
        if plot_type == "Pair Plot":
            fig = sns.pairplot(df[selected_columns].dropna())
            st.pyplot(fig)
        elif plot_type == "Scatter Plot":
            x_axis = st.selectbox("📍 X-axis", selected_columns, index=0)
            y_axis = st.selectbox("📍 Y-axis", selected_columns, index=1)
            hue_option = None
            if 'Outlier' in df.columns:
                # Ensure 'Outlier' is boolean and has no NaNs
                df['Outlier'] = df['Outlier'].fillna(False).astype(bool)
                hue_option = 'Outlier'
            fig, ax = plt.subplots(figsize=(10, 6))
            if hue_option:
                # Check if 'Outlier' column has more than one unique value to avoid warning
                if df[hue_option].nunique() > 1:
                    sns.scatterplot(
                        data=df, 
                        x=x_axis, 
                        y=y_axis, 
                        hue=hue_option,
                        palette={True: 'red', False: 'blue'},
                        ax=ax,
                        alpha=0.7
                    )
                else:
                    # If 'Outlier' has only one unique value, skip hue
                    sns.scatterplot(
                        data=df, 
                        x=x_axis, 
                        y=y_axis, 
                        ax=ax,
                        color='blue',
                        alpha=0.7
                    )
                    st.warning("⚠️ 'Outlier' column has only one unique value. Hue parameter is ignored.")
            else:
                sns.scatterplot(
                    data=df, 
                    x=x_axis, 
                    y=y_axis, 
                    ax=ax,
                    color='blue',
                    alpha=0.7
                )
            ax.set_title(f"Scatter Plot of `{x_axis}` vs `{y_axis}`", fontsize=16)
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please select at least two numerical columns for multivariate visualization.")

def choose_statistical_test():
    """
    Allows users to select a statistical test for outlier detection.

    Returns:
        str: The name of the selected statistical test.
    """
    st.sidebar.header("🔍 Outlier Detection")
    test = st.sidebar.selectbox(
        "🛠️ Select a statistical test to detect outliers", 
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
    st.header("🔎 Outlier Identification")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("⚠️ No numerical columns available for outlier detection.")
        return df
    selected_columns = st.multiselect(
        "📋 Select numerical columns for outlier detection", 
        numeric_columns, 
        default=numeric_columns[:2]
    )
    
    if selected_columns:
        if test == "Z-Score":
            threshold = st.sidebar.number_input("🔢 Z-Score Threshold", value=3.0, step=0.1)
            # Calculate z-scores
            z_scores = df[selected_columns].apply(lambda x: np.abs(stats.zscore(x, nan_policy='omit')))
            # Identify outliers
            outliers = z_scores > threshold
            # Any row with any outlier in selected columns
            outlier_mask = outliers.any(axis=1)
        elif test == "IQR":
            multiplier = st.sidebar.number_input("📏 IQR Multiplier", value=1.5, step=0.1)
            Q1 = df[selected_columns].quantile(0.25)
            Q3 = df[selected_columns].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            # Identify outliers
            outliers = (df[selected_columns] < lower_bound) | (df[selected_columns] > upper_bound)
            # Any row with any outlier in selected columns
            outlier_mask = outliers.any(axis=1)
        elif test == "DBSCAN":
            eps = st.sidebar.number_input("🔧 DBSCAN eps", value=0.5, step=0.1)
            min_samples = st.sidebar.number_input("🔧 DBSCAN min_samples", value=5, step=1)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[selected_columns].dropna())
            db = DBSCAN(eps=eps, min_samples=min_samples)
            db.fit(scaled_data)
            labels = db.labels_
            outliers_detected = labels == -1
            # Initialize a Series with all False
            outlier_mask = pd.Series(False, index=df.index)
            # Assign outlier status to the corresponding indices
            outlier_mask.loc[df[selected_columns].dropna().index] = outliers_detected
        elif test == "Isolation Forest":
            contamination = st.sidebar.number_input(
                "🧮 Isolation Forest Contamination",
                value=0.05, 
                min_value=0.0, 
                max_value=0.5, 
                step=0.01
            )
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            # Fit the model
            iso_forest.fit(df[selected_columns].dropna())
            preds = iso_forest.predict(df[selected_columns].dropna())
            outliers_detected = preds == -1
            # Initialize a Series with all False
            outlier_mask = pd.Series(False, index=df.index)
            # Assign outlier status to the corresponding indices
            outlier_mask.loc[df[selected_columns].dropna().index] = outliers_detected

        # Add the 'Outlier' column
        df['Outlier'] = outlier_mask.fillna(False).astype(bool)
        num_outliers = df['Outlier'].sum()
        st.write(f"**Number of outliers detected:** {num_outliers}")
        
        # Optionally, display some statistics or a table of outliers
        if num_outliers > 0:
            st.subheader("📋 Outlier Records")
            st.write(df[df['Outlier']])
    else:
        st.warning("⚠️ Please select at least one numerical column for outlier detection.")
    return df

def download_enhanced_dataset(df):
    """
    Provides an option for users to download the enhanced dataset with outlier flags.

    Args:
        df (pd.DataFrame): The enhanced dataset.
    """
    st.header("💾 Download Enhanced Dataset")
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.save()
        processed_data = output.getvalue()
        return processed_data

    excel_data = to_excel(df)
    st.download_button(
        label="📥 Download dataset with Outlier Flags",
        data=excel_data,
        file_name='enhanced_dataset.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

def main():
    """
    The main function that orchestrates the Streamlit application.
    """
    st.set_page_config(
        page_title="📊 Outlier Detection and Analysis App",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📈 Outlier Detection and Analysis App")
    st.markdown("""
    Welcome to the **Outlier Detection and Analysis App**! This application allows you to:
    - 📥 **Upload** your dataset (CSV or Excel).
    - 🔍 **Filter** data based on custom conditions.
    - 🧮 **Aggregate** scores from multiple numerical columns.
    - 📊 **Visualize** data distributions (monovariate and multivariate).
    - 🔎 **Detect** outliers using various statistical methods.
    - 💾 **Download** the enhanced dataset with outlier flags.
    
    **Instructions:**
    1. **Upload** your dataset using the upload button.
    2. **Filter** the data by entering your conditions in the filter section.
    3. **Aggregate** scores if needed.
    4. **Visualize** the data distributions.
    5. **Choose** a statistical test to detect outliers.
    6. **Download** the enhanced dataset with outlier information.
    """)

    # Step 1: Upload Dataset
    df = upload_dataset()
    if df is not None:
        st.markdown("---")
        # Step 2: Filter Cases
        filtered_df = filter_cases(df)
        
        st.markdown("---")
        # Step 3: Aggregate Scores
        aggregated_df = aggregate_scores(filtered_df)
        
        st.markdown("---")
        # Step 4: Visualizations
        st.header("📊 Data Visualization")
        col1, col2 = st.columns(2)
        with col1:
            plot_monovariate_distribution(aggregated_df)
        with col2:
            plot_multivariate_distribution(aggregated_df)
        
        st.markdown("---")
        # Step 5: Choose Statistical Test
        selected_test = choose_statistical_test()
        
        # Step 6: Identify Outliers
        outlier_df = identify_outliers(aggregated_df, selected_test)
        
        st.markdown("---")
        # Step 7: Download Enhanced Dataset
        download_enhanced_dataset(outlier_df)
        
        st.markdown("---")
        # Optional: Display the enhanced dataset
        st.header("📂 Enhanced Dataset")
        st.dataframe(outlier_df)
        
        st.subheader("📊 Dataset Statistics")
        st.write(outlier_df.describe())

if __name__ == "__main__":
    main()
