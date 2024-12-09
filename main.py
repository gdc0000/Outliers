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

def add_footer():
    """
    Adds a footer with professional information and links.
    """
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown("""
    [GitHub](https://github.com/gdc0000) | 
    [ORCID](https://orcid.org/0000-0002-1439-5790) | 
    [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """)

def upload_dataset():
    """
    Allows users to upload a dataset in CSV, Excel, or other supported formats.

    Returns:
        pd.DataFrame: The uploaded dataset as a pandas DataFrame.
    """
    st.sidebar.header("📥 **Upload Dataset**")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success("✅ Dataset uploaded successfully!")
            st.sidebar.markdown(f"**Dataset Shape:** {df.shape}")
            st.sidebar.write("**Preview:**")
            st.sidebar.dataframe(df.head())
            return df
        except Exception as e:
            st.sidebar.error(f"❌ Error loading dataset: {e}")
    return None

def filter_cases(df):
    """
    Provides functionality to filter instances (rows) based on user-defined conditions.

    Args:
        df (pd.DataFrame): The original dataset.

    Returns:
        pd.DataFrame: The filtered dataset.
    """
    st.header("🔍 **Filter Cases**")
    
    st.markdown("""
    ### Apply Filter Conditions
    Enter your filter conditions using pandas `query` syntax.

    - **Logical Operators:** `and`, `or`, `not`
    - **Comparison Operators:** `>`, `<`, `>=`, `<=`, `==`, `!=`
    - **String Values:** Enclose string literals in single (`'`) or double (`"`) quotes.

    **Examples:**
    - `AGE > 30`
    - `INCOME < 50000 and AGE >= 25`
    - `CITY == 'New York' or CITY == 'Los Angeles'`
    """)
    
    with st.form("filter_form"):
        filter_condition = st.text_area(
            "📋 **Enter Filter Conditions**",
            height=150,
            placeholder="e.g., AGE > 30 and INCOME < 50000"
        )
        submitted = st.form_submit_button("🔄 **Apply Filter**")
        if submitted:
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
    st.write(f"**Number of records after filtering:** {df.shape[0]}")
    return df

def aggregate_scores(df):
    """
    Enables users to create aggregated scores by selecting numerical columns and assigning custom names.

    Args:
        df (pd.DataFrame): The dataset to aggregate.

    Returns:
        pd.DataFrame: Dataset with aggregated score columns.
    """
    st.header("🧮 **Score Aggregation & Variable Transformation**")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("⚠️ No numerical columns available for aggregation.")
        return df

    with st.form("aggregation_form"):
        st.subheader("📊 **Select Columns to Aggregate**")
        selected_columns = st.multiselect(
            "Select Columns", 
            numeric_columns,
            default=numeric_columns[:2]
        )
        aggregation_method = st.selectbox(
            "🔧 **Select Aggregation Method**", 
            ["Sum", "Mean", "Weighted Sum"]
        )
        # Custom name for aggregated score
        if aggregation_method != "Weighted Sum":
            agg_custom_name = st.text_input(
                "✏️ **Enter Custom Name for Aggregated Score**",
                value=f"Composite_Score_{aggregation_method}"
            )
        else:
            agg_custom_name = st.text_input(
                "✏️ **Enter Custom Name for Weighted Sum Score**",
                value="Composite_Score_Weighted_Sum"
            )
        # Variable Transformation
        st.subheader("🔄 **Variable Transformation**")
        transformation = st.selectbox(
            "Select Variable Transformation",
            ["None", "Log Transformation", "Square Root Transformation", "Custom Expression"]
        )
        # Custom name for transformed variable
        if transformation != "None":
            trans_custom_name = st.text_input(
                "✏️ **Enter Custom Name for Transformed Variable**",
                value=f"{transformation}_Transformed"
            )
        else:
            trans_custom_name = ""
        # For custom expression
        custom_expression = ""
        if transformation == "Custom Expression":
            custom_expression = st.text_input(
                "✏️ **Enter Custom Expression**",
                help="Use pandas syntax. Example: `df['New_Column'] = df['A'] / df['B']`"
            )
        submitted = st.form_submit_button("🧮 **Apply Aggregation & Transformation**")
        
        if submitted:
            # Aggregation
            if selected_columns:
                if aggregation_method == "Sum":
                    if agg_custom_name in df.columns:
                        st.warning(f"⚠️ Column name `{agg_custom_name}` already exists. It will be overwritten.")
                    df[agg_custom_name] = df[selected_columns].sum(axis=1)
                    st.success(f"✅ Aggregated score (Sum) created as `{agg_custom_name}`.")
                elif aggregation_method == "Mean":
                    if agg_custom_name in df.columns:
                        st.warning(f"⚠️ Column name `{agg_custom_name}` already exists. It will be overwritten.")
                    df[agg_custom_name] = df[selected_columns].mean(axis=1)
                    st.success(f"✅ Aggregated score (Mean) created as `{agg_custom_name}`.")
                elif aggregation_method == "Weighted Sum":
                    weights = {}
                    st.markdown("### ⚖️ **Assign Weights**")
                    for col in selected_columns:
                        weights[col] = st.number_input(
                            f"Weight for `{col}`", 
                            value=1.0, 
                            step=0.1,
                            format="%.2f"
                        )
                    weights_series = pd.Series(weights)
                    if agg_custom_name in df.columns:
                        st.warning(f"⚠️ Column name `{agg_custom_name}` already exists. It will be overwritten.")
                    df[agg_custom_name] = df[selected_columns].mul(weights_series).sum(axis=1)
                    st.success(f"✅ Aggregated score (Weighted Sum) created as `{agg_custom_name}`.")
                st.write("### 📈 **Aggregated Score Statistics**")
                st.write(df[[agg_custom_name]].describe())
            else:
                st.warning("⚠️ Please select at least one column to aggregate.")
            
            # Variable Transformation
            if transformation != "None":
                if transformation == "Log Transformation":
                    selected_transform = st.selectbox(
                        "📌 **Select Column for Log Transformation**",
                        numeric_columns
                    )
                    if selected_transform:
                        # Handle non-positive values
                        if (df[selected_transform] <= 0).any():
                            st.error(f"❌ Cannot apply log transformation on `{selected_transform}` with non-positive values.")
                        else:
                            if trans_custom_name in df.columns:
                                st.warning(f"⚠️ Column name `{trans_custom_name}` already exists. It will be overwritten.")
                            df[trans_custom_name] = np.log(df[selected_transform])
                            st.success(f"✅ Log transformation applied to `{selected_transform}` as `{trans_custom_name}`.")
                elif transformation == "Square Root Transformation":
                    selected_transform = st.selectbox(
                        "📌 **Select Column for Square Root Transformation**",
                        numeric_columns
                    )
                    if selected_transform:
                        # Handle negative values
                        if (df[selected_transform] < 0).any():
                            st.error(f"❌ Cannot apply square root transformation on `{selected_transform}` with negative values.")
                        else:
                            if trans_custom_name in df.columns:
                                st.warning(f"⚠️ Column name `{trans_custom_name}` already exists. It will be overwritten.")
                            df[trans_custom_name] = np.sqrt(df[selected_transform])
                            st.success(f"✅ Square root transformation applied to `{selected_transform}` as `{trans_custom_name}`.")
                elif transformation == "Custom Expression":
                    if custom_expression.strip() == "":
                        st.warning("⚠️ Please enter a valid custom expression.")
                    else:
                        try:
                            # Execute the custom expression safely
                            # Note: Using exec can be dangerous; ensure only trusted inputs are allowed
                            exec(custom_expression, {'df': df, 'np': np, 'pd': pd})
                            st.success("✅ Custom transformation applied successfully.")
                        except Exception as e:
                            st.error(f"❌ Error in custom transformation: {e}")
            st.write("### 📈 **Updated Dataset Preview**")
            st.dataframe(df.head())
    return df

def plot_monovariate_distribution(df):
    """
    Visualizes the distribution of individual numerical variables.

    Args:
        df (pd.DataFrame): The dataset to visualize.
    """
    st.header("📊 **Monovariate Distribution**")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("⚠️ No numerical columns available for visualization.")
        return
    with st.form("monovariate_form"):
        selected_column = st.selectbox("🔍 **Select a Numerical Column to Visualize**", numeric_columns)
        plot_type = st.selectbox("🔧 **Select Plot Type**", ["Histogram & KDE", "Box Plot", "Violin Plot"])
        submitted = st.form_submit_button("📈 **Generate Plot**")
        if submitted:
            if selected_column:
                fig, ax = plt.subplots(figsize=(10, 6))
                if plot_type == "Histogram & KDE":
                    sns.histplot(df[selected_column].dropna(), kde=True, ax=ax, color='skyblue', edgecolor='black')
                    ax.set_title(f"📈 **Distribution of `{selected_column}`**", fontsize=16)
                    ax.set_xlabel(selected_column, fontsize=14)
                    ax.set_ylabel("Frequency", fontsize=14)
                elif plot_type == "Box Plot":
                    sns.boxplot(y=df[selected_column], ax=ax, color='lightgreen')
                    ax.set_title(f"📦 **Box Plot of `{selected_column}`**", fontsize=16)
                    ax.set_ylabel(selected_column, fontsize=14)
                elif plot_type == "Violin Plot":
                    sns.violinplot(y=df[selected_column], ax=ax, color='lightcoral')
                    ax.set_title(f"🎻 **Violin Plot of `{selected_column}`**", fontsize=16)
                    ax.set_ylabel(selected_column, fontsize=14)
                st.pyplot(fig)

def plot_multivariate_distribution(df):
    """
    Visualizes the distribution of multiple numerical variables using pair plots, scatter plots, or correlation heatmaps.

    Args:
        df (pd.DataFrame): The dataset to visualize.
    """
    st.header("📈 **Multivariate Distribution**")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_columns) < 2:
        st.warning("⚠️ At least two numerical columns are required for multivariate visualization.")
        return
    with st.form("multivariate_form"):
        selected_columns = st.multiselect(
            "🔍 **Select Numerical Columns to Visualize**", 
            numeric_columns, 
            default=numeric_columns[:2]
        )
        
        plot_type = st.selectbox("🔧 **Select Plot Type**", ["Pair Plot", "Scatter Plot", "Correlation Heatmap"])
        submitted = st.form_submit_button("📊 **Generate Plot**")
        if submitted:
            if len(selected_columns) >= 2:
                if plot_type == "Pair Plot":
                    with st.spinner("Generating Pair Plot..."):
                        fig = sns.pairplot(df[selected_columns].dropna())
                        st.pyplot(fig)
                elif plot_type == "Scatter Plot":
                    x_axis = st.selectbox("📍 **X-axis**", selected_columns, index=0)
                    y_axis = st.selectbox("📍 **Y-axis**", selected_columns, index=1)
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
                    ax.set_title(f"📉 **Scatter Plot of `{x_axis}` vs `{y_axis}`**", fontsize=16)
                    st.pyplot(fig)
                elif plot_type == "Correlation Heatmap":
                    with st.spinner("Generating Correlation Heatmap..."):
                        corr = df[selected_columns].corr()
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                        ax.set_title("📈 **Correlation Heatmap**", fontsize=16)
                        st.pyplot(fig)
            else:
                st.warning("⚠️ Please select at least two numerical columns for multivariate visualization.")

def choose_statistical_test():
    """
    Allows users to select a statistical test for outlier detection.

    Returns:
        str: The name of the selected statistical test.
    """
    st.sidebar.header("🔍 **Outlier Detection Settings**")
    test = st.sidebar.selectbox(
        "🛠️ **Select a Statistical Test to Detect Outliers**", 
        ["Z-Score", "IQR", "DBSCAN", "Isolation Forest"]
    )
    
    # Educational Information about the selected test
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 **About the Selected Test**")
    
    if test == "Z-Score":
        st.sidebar.markdown("""
        **Z-Score Method**
        
        - **Description:** Measures how many standard deviations an element is from the mean.
        - **Usage:** Suitable for data with a normal distribution.
        - **Outlier Criteria:** Typically, a Z-Score above 3 or below -3 is considered an outlier.
        """)
    elif test == "IQR":
        st.sidebar.markdown("""
        **Interquartile Range (IQR) Method**
        
        - **Description:** Uses the spread of the middle 50% of the data to identify outliers.
        - **Usage:** Suitable for data with skewed distributions.
        - **Outlier Criteria:** Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are considered outliers.
        """)
    elif test == "DBSCAN":
        st.sidebar.markdown("""
        **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
        
        - **Description:** A clustering algorithm that identifies outliers as points not belonging to any cluster.
        - **Usage:** Suitable for data with clusters of similar density.
        - **Parameters:**
          - **eps:** Maximum distance between two samples for them to be considered in the same neighborhood.
          - **min_samples:** Minimum number of samples in a neighborhood to form a core point.
        """)
    elif test == "Isolation Forest":
        st.sidebar.markdown("""
        **Isolation Forest**
        
        - **Description:** An ensemble-based algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
        - **Usage:** Effective for high-dimensional datasets.
        - **Parameters:**
          - **contamination:** The proportion of outliers in the data set.
        """)
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
    st.header("🔎 **Outlier Identification**")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("⚠️ No numerical columns available for outlier detection.")
        return df
    with st.form("outlier_form"):
        selected_columns = st.multiselect(
            "📋 **Select Numerical Columns for Outlier Detection**", 
            numeric_columns, 
            default=numeric_columns[:2]
        )
        
        submitted = st.form_submit_button("🔍 **Detect Outliers**")
        if submitted:
            if selected_columns:
                if test == "Z-Score":
                    threshold = st.sidebar.number_input("🔢 **Z-Score Threshold**", value=3.0, step=0.1)
                    # Calculate z-scores, handling NaNs
                    z_scores = df[selected_columns].apply(lambda x: np.abs(stats.zscore(x, nan_policy='omit')))
                    # Identify outliers
                    outliers = z_scores > threshold
                    # Any row with any outlier in selected columns
                    outlier_mask = outliers.any(axis=1)
                elif test == "IQR":
                    multiplier = st.sidebar.number_input("📏 **IQR Multiplier**", value=1.5, step=0.1)
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
                    eps = st.sidebar.number_input("🔧 **DBSCAN eps**", value=0.5, step=0.1)
                    min_samples = st.sidebar.number_input("🔧 **DBSCAN min_samples**", value=5, step=1)
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
                        "🧮 **Isolation Forest Contamination**",
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

                # Add the 'Outlier' column, ensuring no NaNs
                df['Outlier'] = outlier_mask.fillna(False).astype(bool)
                num_outliers = df['Outlier'].sum()
                st.write(f"**Number of outliers detected:** {num_outliers}")
                
                # Optionally, display some statistics or a table of outliers
                if num_outliers > 0:
                    st.subheader("📋 **Outlier Records**")
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
    st.header("💾 **Download Enhanced Dataset**")
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            # No need to call writer.save(); it's handled by the context manager
        processed_data = output.getvalue()
        return processed_data

    excel_data = to_excel(df)
    st.download_button(
        label="📥 **Download Dataset with Outlier Flags**",
        data=excel_data,
        file_name='enhanced_dataset.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

def remove_columns(df):
    """
    Allows users to remove unwanted columns from the dataset.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: Dataset with selected columns removed.
    """
    st.header("🗑️ **Remove Columns**")
    all_columns = df.columns.tolist()
    with st.form("remove_columns_form"):
        columns_to_remove = st.multiselect(
            "🚮 **Select Columns to Remove**",
            all_columns,
            help="Select one or more columns to remove from the dataset."
        )
        submitted = st.form_submit_button("🗑️ **Remove Selected Columns**")
        if submitted:
            if columns_to_remove:
                df = df.drop(columns=columns_to_remove)
                st.success(f"✅ Removed columns: {', '.join(columns_to_remove)}")
                st.write(f"**Updated Dataset Shape:** {df.shape}")
                st.dataframe(df.head())
            else:
                st.warning("⚠️ Please select at least one column to remove.")
    return df

def add_custom_variables(df):
    """
    Allows users to add multiple aggregated scores or transformed variables with custom names.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: Dataset with additional custom variables.
    """
    st.header("➕ **Add Multiple Aggregated Scores & Variable Transformations**")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("⚠️ No numerical columns available for aggregation or transformation.")
        return df

    num_operations = st.number_input(
        "🔢 **Number of Operations to Add**",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Specify how many aggregation or transformation operations you want to add."
    )

    for i in range(int(num_operations)):
        st.subheader(f"🛠️ **Operation {i+1}**")
        operation_type = st.selectbox(
            f"🔧 **Select Operation Type for Operation {i+1}**",
            ["Aggregation", "Variable Transformation"],
            key=f"operation_type_{i}"
        )
        if operation_type == "Aggregation":
            selected_columns = st.multiselect(
                f"📊 **Select Columns to Aggregate for Operation {i+1}**", 
                numeric_columns,
                key=f"agg_columns_{i}"
            )
            aggregation_method = st.selectbox(
                f"🔧 **Select Aggregation Method for Operation {i+1}**", 
                ["Sum", "Mean", "Weighted Sum"],
                key=f"agg_method_{i}"
            )
            # Custom name for aggregated score
            agg_custom_name = st.text_input(
                f"✏️ **Enter Custom Name for Aggregated Score Operation {i+1}**",
                value=f"Custom_Agg_Score_{i+1}",
                key=f"agg_name_{i}"
            )
            if aggregation_method == "Weighted Sum":
                st.markdown(f"### ⚖️ **Assign Weights for Operation {i+1}**")
                weights = {}
                for col in selected_columns:
                    weights[col] = st.number_input(
                        f"Weight for `{col}`", 
                        value=1.0, 
                        step=0.1,
                        format="%.2f",
                        key=f"weight_{i}_{col}"
                    )
                weights_series = pd.Series(weights)
            else:
                weights_series = None

            apply_agg = st.button(f"✅ **Apply Aggregation Operation {i+1}**", key=f"apply_agg_{i}")
            if apply_agg:
                if selected_columns:
                    if aggregation_method == "Sum":
                        df[agg_custom_name] = df[selected_columns].sum(axis=1)
                        st.success(f"✅ Aggregated score (Sum) created as `{agg_custom_name}`.")
                    elif aggregation_method == "Mean":
                        df[agg_custom_name] = df[selected_columns].mean(axis=1)
                        st.success(f"✅ Aggregated score (Mean) created as `{agg_custom_name}`.")
                    elif aggregation_method == "Weighted Sum":
                        df[agg_custom_name] = df[selected_columns].mul(weights_series).sum(axis=1)
                        st.success(f"✅ Aggregated score (Weighted Sum) created as `{agg_custom_name}`.")
                    st.write("### 📈 **Aggregated Score Statistics**")
                    st.write(df[[agg_custom_name]].describe())
                else:
                    st.warning("⚠️ Please select at least one column to aggregate.")
        
        elif operation_type == "Variable Transformation":
            transformation = st.selectbox(
                f"🔄 **Select Variable Transformation for Operation {i+1}**",
                ["None", "Log Transformation", "Square Root Transformation", "Custom Expression"],
                key=f"transformation_{i}"
            )
            if transformation != "None":
                trans_custom_name = st.text_input(
                    f"✏️ **Enter Custom Name for Transformed Variable Operation {i+1}**",
                    value=f"Custom_Transformed_{i+1}",
                    key=f"trans_name_{i}"
                )
            else:
                trans_custom_name = ""
            # For custom expression
            custom_expression = ""
            if transformation == "Custom Expression":
                custom_expression = st.text_input(
                    f"✏️ **Enter Custom Expression for Operation {i+1}**",
                    help="Use pandas syntax. Example: `df['New_Column'] = df['A'] / df['B']`",
                    key=f"custom_expr_{i}"
                )
            apply_trans = st.button(f"✅ **Apply Transformation Operation {i+1}**", key=f"apply_trans_{i}")
            if apply_trans:
                if transformation == "Log Transformation":
                    selected_transform = st.selectbox(
                        f"📌 **Select Column for Log Transformation Operation {i+1}**",
                        numeric_columns,
                        key=f"log_col_{i}"
                    )
                    if selected_transform:
                        # Handle non-positive values
                        if (df[selected_transform] <= 0).any():
                            st.error(f"❌ Cannot apply log transformation on `{selected_transform}` with non-positive values.")
                        else:
                            df[trans_custom_name] = np.log(df[selected_transform])
                            st.success(f"✅ Log transformation applied to `{selected_transform}` as `{trans_custom_name}`.")
                elif transformation == "Square Root Transformation":
                    selected_transform = st.selectbox(
                        f"📌 **Select Column for Square Root Transformation Operation {i+1}**",
                        numeric_columns,
                        key=f"sqrt_col_{i}"
                    )
                    if selected_transform:
                        # Handle negative values
                        if (df[selected_transform] < 0).any():
                            st.error(f"❌ Cannot apply square root transformation on `{selected_transform}` with negative values.")
                        else:
                            df[trans_custom_name] = np.sqrt(df[selected_transform])
                            st.success(f"✅ Square root transformation applied to `{selected_transform}` as `{trans_custom_name}`.")
                elif transformation == "Custom Expression":
                    if custom_expression.strip() == "":
                        st.warning("⚠️ Please enter a valid custom expression.")
                    else:
                        try:
                            # Execute the custom expression safely
                            # Note: Using exec can be dangerous; ensure only trusted inputs are allowed
                            exec(custom_expression, {'df': df, 'np': np, 'pd': pd})
                            st.success("✅ Custom transformation applied successfully.")
                        except Exception as e:
                            st.error(f"❌ Error in custom transformation: {e}")
            st.write("### 📈 **Updated Dataset Preview**")
            st.dataframe(df.head())
    return df

def plot_monovariate_distribution(df):
    """
    Visualizes the distribution of individual numerical variables.

    Args:
        df (pd.DataFrame): The dataset to visualize.
    """
    st.header("📊 **Monovariate Distribution**")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("⚠️ No numerical columns available for visualization.")
        return
    with st.form("monovariate_form"):
        selected_column = st.selectbox("🔍 **Select a Numerical Column to Visualize**", numeric_columns)
        plot_type = st.selectbox("🔧 **Select Plot Type**", ["Histogram & KDE", "Box Plot", "Violin Plot"])
        submitted = st.form_submit_button("📈 **Generate Plot**")
        if submitted:
            if selected_column:
                fig, ax = plt.subplots(figsize=(10, 6))
                if plot_type == "Histogram & KDE":
                    sns.histplot(df[selected_column].dropna(), kde=True, ax=ax, color='skyblue', edgecolor='black')
                    ax.set_title(f"📈 **Distribution of `{selected_column}`**", fontsize=16)
                    ax.set_xlabel(selected_column, fontsize=14)
                    ax.set_ylabel("Frequency", fontsize=14)
                elif plot_type == "Box Plot":
                    sns.boxplot(y=df[selected_column], ax=ax, color='lightgreen')
                    ax.set_title(f"📦 **Box Plot of `{selected_column}`**", fontsize=16)
                    ax.set_ylabel(selected_column, fontsize=14)
                elif plot_type == "Violin Plot":
                    sns.violinplot(y=df[selected_column], ax=ax, color='lightcoral')
                    ax.set_title(f"🎻 **Violin Plot of `{selected_column}`**", fontsize=16)
                    ax.set_ylabel(selected_column, fontsize=14)
                st.pyplot(fig)

def plot_multivariate_distribution(df):
    """
    Visualizes the distribution of multiple numerical variables using pair plots, scatter plots, or correlation heatmaps.

    Args:
        df (pd.DataFrame): The dataset to visualize.
    """
    st.header("📈 **Multivariate Distribution**")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_columns) < 2:
        st.warning("⚠️ At least two numerical columns are required for multivariate visualization.")
        return
    with st.form("multivariate_form"):
        selected_columns = st.multiselect(
            "🔍 **Select Numerical Columns to Visualize**", 
            numeric_columns, 
            default=numeric_columns[:2]
        )
        
        plot_type = st.selectbox("🔧 **Select Plot Type**", ["Pair Plot", "Scatter Plot", "Correlation Heatmap"])
        submitted = st.form_submit_button("📊 **Generate Plot**")
        if submitted:
            if len(selected_columns) >= 2:
                if plot_type == "Pair Plot":
                    with st.spinner("Generating Pair Plot..."):
                        fig = sns.pairplot(df[selected_columns].dropna())
                        st.pyplot(fig)
                elif plot_type == "Scatter Plot":
                    x_axis = st.selectbox("📍 **X-axis**", selected_columns, key="scatter_x")
                    y_axis = st.selectbox("📍 **Y-axis**", selected_columns, key="scatter_y")
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
                    ax.set_title(f"📉 **Scatter Plot of `{x_axis}` vs `{y_axis}`**", fontsize=16)
                    st.pyplot(fig)
                elif plot_type == "Correlation Heatmap":
                    with st.spinner("Generating Correlation Heatmap..."):
                        corr = df[selected_columns].corr()
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                        ax.set_title("📈 **Correlation Heatmap**", fontsize=16)
                        st.pyplot(fig)
            else:
                st.warning("⚠️ Please select at least two numerical columns for multivariate visualization.")

def choose_statistical_test():
    """
    Allows users to select a statistical test for outlier detection.

    Returns:
        str: The name of the selected statistical test.
    """
    st.sidebar.header("🔍 **Outlier Detection Settings**")
    test = st.sidebar.selectbox(
        "🛠️ **Select a Statistical Test to Detect Outliers**", 
        ["Z-Score", "IQR", "DBSCAN", "Isolation Forest"]
    )
    
    # Educational Information about the selected test
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 **About the Selected Test**")
    
    if test == "Z-Score":
        st.sidebar.markdown("""
        **Z-Score Method**
        
        - **Description:** Measures how many standard deviations an element is from the mean.
        - **Usage:** Suitable for data with a normal distribution.
        - **Outlier Criteria:** Typically, a Z-Score above 3 or below -3 is considered an outlier.
        """)
    elif test == "IQR":
        st.sidebar.markdown("""
        **Interquartile Range (IQR) Method**
        
        - **Description:** Uses the spread of the middle 50% of the data to identify outliers.
        - **Usage:** Suitable for data with skewed distributions.
        - **Outlier Criteria:** Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are considered outliers.
        """)
    elif test == "DBSCAN":
        st.sidebar.markdown("""
        **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
        
        - **Description:** A clustering algorithm that identifies outliers as points not belonging to any cluster.
        - **Usage:** Suitable for data with clusters of similar density.
        - **Parameters:**
          - **eps:** Maximum distance between two samples for them to be considered in the same neighborhood.
          - **min_samples:** Minimum number of samples in a neighborhood to form a core point.
        """)
    elif test == "Isolation Forest":
        st.sidebar.markdown("""
        **Isolation Forest**
        
        - **Description:** An ensemble-based algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
        - **Usage:** Effective for high-dimensional datasets.
        - **Parameters:**
          - **contamination:** The proportion of outliers in the data set.
        """)
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
    st.header("🔎 **Outlier Identification**")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("⚠️ No numerical columns available for outlier detection.")
        return df
    with st.form("outlier_form"):
        selected_columns = st.multiselect(
            "📋 **Select Numerical Columns for Outlier Detection**", 
            numeric_columns, 
            default=numeric_columns[:2]
        )
        
        submitted = st.form_submit_button("🔍 **Detect Outliers**")
        if submitted:
            if selected_columns:
                if test == "Z-Score":
                    threshold = st.sidebar.number_input("🔢 **Z-Score Threshold**", value=3.0, step=0.1)
                    # Calculate z-scores, handling NaNs
                    z_scores = df[selected_columns].apply(lambda x: np.abs(stats.zscore(x, nan_policy='omit')))
                    # Identify outliers
                    outliers = z_scores > threshold
                    # Any row with any outlier in selected columns
                    outlier_mask = outliers.any(axis=1)
                elif test == "IQR":
                    multiplier = st.sidebar.number_input("📏 **IQR Multiplier**", value=1.5, step=0.1)
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
                    eps = st.sidebar.number_input("🔧 **DBSCAN eps**", value=0.5, step=0.1)
                    min_samples = st.sidebar.number_input("🔧 **DBSCAN min_samples**", value=5, step=1)
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
                        "🧮 **Isolation Forest Contamination**",
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

                # Add the 'Outlier' column, ensuring no NaNs
                df['Outlier'] = outlier_mask.fillna(False).astype(bool)
                num_outliers = df['Outlier'].sum()
                st.write(f"**Number of outliers detected:** {num_outliers}")
                
                # Optionally, display some statistics or a table of outliers
                if num_outliers > 0:
                    st.subheader("📋 **Outlier Records**")
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
    st.header("💾 **Download Enhanced Dataset**")
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            # No need to call writer.save(); it's handled by the context manager
        processed_data = output.getvalue()
        return processed_data

    excel_data = to_excel(df)
    st.download_button(
        label="📥 **Download Dataset with Outlier Flags**",
        data=excel_data,
        file_name='enhanced_dataset.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

def remove_columns(df):
    """
    Allows users to remove unwanted columns from the dataset.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: Dataset with selected columns removed.
    """
    st.header("🗑️ **Remove Columns**")
    all_columns = df.columns.tolist()
    with st.form("remove_columns_form"):
        columns_to_remove = st.multiselect(
            "🚮 **Select Columns to Remove**",
            all_columns,
            help="Select one or more columns to remove from the dataset."
        )
        submitted = st.form_submit_button("🗑️ **Remove Selected Columns**")
        if submitted:
            if columns_to_remove:
                df = df.drop(columns=columns_to_remove)
                st.success(f"✅ Removed columns: {', '.join(columns_to_remove)}")
                st.write(f"**Updated Dataset Shape:** {df.shape}")
                st.dataframe(df.head())
            else:
                st.warning("⚠️ Please select at least one column to remove.")
    return df

def add_custom_variables(df):
    """
    Allows users to add multiple aggregated scores or transformed variables with custom names.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: Dataset with additional custom variables.
    """
    st.header("➕ **Add Multiple Aggregated Scores & Variable Transformations**")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("⚠️ No numerical columns available for aggregation or transformation.")
        return df

    num_operations = st.number_input(
        "🔢 **Number of Operations to Add**",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Specify how many aggregation or transformation operations you want to add."
    )

    for i in range(int(num_operations)):
        st.subheader(f"🛠️ **Operation {i+1}**")
        operation_type = st.selectbox(
            f"🔧 **Select Operation Type for Operation {i+1}**",
            ["Aggregation", "Variable Transformation"],
            key=f"operation_type_{i}"
        )
        if operation_type == "Aggregation":
            selected_columns = st.multiselect(
                f"📊 **Select Columns to Aggregate for Operation {i+1}**", 
                numeric_columns,
                key=f"agg_columns_{i}"
            )
            aggregation_method = st.selectbox(
                f"🔧 **Select Aggregation Method for Operation {i+1}**", 
                ["Sum", "Mean", "Weighted Sum"],
                key=f"agg_method_{i}"
            )
            # Custom name for aggregated score
            agg_custom_name = st.text_input(
                f"✏️ **Enter Custom Name for Aggregated Score Operation {i+1}**",
                value=f"Custom_Agg_Score_{i+1}",
                key=f"agg_name_{i}"
            )
            if aggregation_method == "Weighted Sum":
                st.markdown(f"### ⚖️ **Assign Weights for Operation {i+1}**")
                weights = {}
                for col in selected_columns:
                    weights[col] = st.number_input(
                        f"Weight for `{col}`", 
                        value=1.0, 
                        step=0.1,
                        format="%.2f",
                        key=f"weight_{i}_{col}"
                    )
                weights_series = pd.Series(weights)
            else:
                weights_series = None

            apply_agg = st.button(f"✅ **Apply Aggregation Operation {i+1}**", key=f"apply_agg_{i}")
            if apply_agg:
                if selected_columns:
                    if aggregation_method == "Sum":
                        if agg_custom_name in df.columns:
                            st.warning(f"⚠️ Column name `{agg_custom_name}` already exists. It will be overwritten.")
                        df[agg_custom_name] = df[selected_columns].sum(axis=1)
                        st.success(f"✅ Aggregated score (Sum) created as `{agg_custom_name}`.")
                    elif aggregation_method == "Mean":
                        if agg_custom_name in df.columns:
                            st.warning(f"⚠️ Column name `{agg_custom_name}` already exists. It will be overwritten.")
                        df[agg_custom_name] = df[selected_columns].mean(axis=1)
                        st.success(f"✅ Aggregated score (Mean) created as `{agg_custom_name}`.")
                    elif aggregation_method == "Weighted Sum":
                        if agg_custom_name in df.columns:
                            st.warning(f"⚠️ Column name `{agg_custom_name}` already exists. It will be overwritten.")
                        df[agg_custom_name] = df[selected_columns].mul(weights_series).sum(axis=1)
                        st.success(f"✅ Aggregated score (Weighted Sum) created as `{agg_custom_name}`.")
                    st.write("### 📈 **Aggregated Score Statistics**")
                    st.write(df[[agg_custom_name]].describe())
                else:
                    st.warning("⚠️ Please select at least one column to aggregate.")
        
        elif operation_type == "Variable Transformation":
            transformation = st.selectbox(
                f"🔄 **Select Variable Transformation for Operation {i+1}**",
                ["None", "Log Transformation", "Square Root Transformation", "Custom Expression"],
                key=f"transformation_{i}"
            )
            if transformation != "None":
                trans_custom_name = st.text_input(
                    f"✏️ **Enter Custom Name for Transformed Variable Operation {i+1}**",
                    value=f"Custom_Transformed_{i+1}",
                    key=f"trans_name_{i}"
                )
            else:
                trans_custom_name = ""
            # For custom expression
            custom_expression = ""
            if transformation == "Custom Expression":
                custom_expression = st.text_input(
                    f"✏️ **Enter Custom Expression for Operation {i+1}**",
                    help="Use pandas syntax. Example: `df['New_Column'] = df['A'] / df['B']`",
                    key=f"custom_expr_{i}"
                )
            apply_trans = st.button(f"✅ **Apply Transformation Operation {i+1}**", key=f"apply_trans_{i}")
            if apply_trans:
                if transformation == "Log Transformation":
                    selected_transform = st.selectbox(
                        f"📌 **Select Column for Log Transformation Operation {i+1}**",
                        numeric_columns,
                        key=f"log_col_{i}"
                    )
                    if selected_transform:
                        # Handle non-positive values
                        if (df[selected_transform] <= 0).any():
                            st.error(f"❌ Cannot apply log transformation on `{selected_transform}` with non-positive values.")
                        else:
                            if trans_custom_name in df.columns:
                                st.warning(f"⚠️ Column name `{trans_custom_name}` already exists. It will be overwritten.")
                            df[trans_custom_name] = np.log(df[selected_transform])
                            st.success(f"✅ Log transformation applied to `{selected_transform}` as `{trans_custom_name}`.")
                elif transformation == "Square Root Transformation":
                    selected_transform = st.selectbox(
                        f"📌 **Select Column for Square Root Transformation Operation {i+1}**",
                        numeric_columns,
                        key=f"sqrt_col_{i}"
                    )
                    if selected_transform:
                        # Handle negative values
                        if (df[selected_transform] < 0).any():
                            st.error(f"❌ Cannot apply square root transformation on `{selected_transform}` with negative values.")
                        else:
                            if trans_custom_name in df.columns:
                                st.warning(f"⚠️ Column name `{trans_custom_name}` already exists. It will be overwritten.")
                            df[trans_custom_name] = np.sqrt(df[selected_transform])
                            st.success(f"✅ Square root transformation applied to `{selected_transform}` as `{trans_custom_name}`.")
                elif transformation == "Custom Expression":
                    if custom_expression.strip() == "":
                        st.warning("⚠️ Please enter a valid custom expression.")
                    else:
                        try:
                            # Execute the custom expression safely
                            # Note: Using exec can be dangerous; ensure only trusted inputs are allowed
                            exec(custom_expression, {'df': df, 'np': np, 'pd': pd})
                            st.success("✅ Custom transformation applied successfully.")
                        except Exception as e:
                            st.error(f"❌ Error in custom transformation: {e}")
            st.write("### 📈 **Updated Dataset Preview**")
            st.dataframe(df.head())
    return df

def plot_monovariate_distribution(df):
    """
    Visualizes the distribution of individual numerical variables.

    Args:
        df (pd.DataFrame): The dataset to visualize.
    """
    st.header("📊 **Monovariate Distribution**")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.warning("⚠️ No numerical columns available for visualization.")
        return
    with st.form("monovariate_form"):
        selected_column = st.selectbox("🔍 **Select a Numerical Column to Visualize**", numeric_columns)
        plot_type = st.selectbox("🔧 **Select Plot Type**", ["Histogram & KDE", "Box Plot", "Violin Plot"])
        submitted = st.form_submit_button("📈 **Generate Plot**")
        if submitted:
            if selected_column:
                fig, ax = plt.subplots(figsize=(10, 6))
                if plot_type == "Histogram & KDE":
                    sns.histplot(df[selected_column].dropna(), kde=True, ax=ax, color='skyblue', edgecolor='black')
                    ax.set_title(f"📈 **Distribution of `{selected_column}`**", fontsize=16)
                    ax.set_xlabel(selected_column, fontsize=14)
                    ax.set_ylabel("Frequency", fontsize=14)
                elif plot_type == "Box Plot":
                    sns.boxplot(y=df[selected_column], ax=ax, color='lightgreen')
                    ax.set_title(f"📦 **Box Plot of `{selected_column}`**", fontsize=16)
                    ax.set_ylabel(selected_column, fontsize=14)
                elif plot_type == "Violin Plot":
                    sns.violinplot(y=df[selected_column], ax=ax, color='lightcoral')
                    ax.set_title(f"🎻 **Violin Plot of `{selected_column}`**", fontsize=16)
                    ax.set_ylabel(selected_column, fontsize=14)
                st.pyplot(fig)

def plot_multivariate_distribution(df):
    """
    Visualizes the distribution of multiple numerical variables using pair plots, scatter plots, or correlation heatmaps.

    Args:
        df (pd.DataFrame): The dataset to visualize.
    """
    st.header("📈 **Multivariate Distribution**")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_columns) < 2:
        st.warning("⚠️ At least two numerical columns are required for multivariate visualization.")
        return
    with st.form("multivariate_form"):
        selected_columns = st.multiselect(
            "🔍 **Select Numerical Columns to Visualize**", 
            numeric_columns, 
            default=numeric_columns[:2]
        )
        
        plot_type = st.selectbox("🔧 **Select Plot Type**", ["Pair Plot", "Scatter Plot", "Correlation Heatmap"])
        submitted = st.form_submit_button("📊 **Generate Plot**")
        if submitted:
            if len(selected_columns) >= 2:
                if plot_type == "Pair Plot":
                    with st.spinner("Generating Pair Plot..."):
                        fig = sns.pairplot(df[selected_columns].dropna())
                        st.pyplot(fig)
                elif plot_type == "Scatter Plot":
                    x_axis = st.selectbox("📍 **X-axis**", selected_columns, key="scatter_x")
                    y_axis = st.selectbox("📍 **Y-axis**", selected_columns, key="scatter_y")
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
                    ax.set_title(f"📉 **Scatter Plot of `{x_axis}` vs `{y_axis}`**", fontsize=16)
                    st.pyplot(fig)
                elif plot_type == "Correlation Heatmap":
                    with st.spinner("Generating Correlation Heatmap..."):
                        corr = df[selected_columns].corr()
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                        ax.set_title("📈 **Correlation Heatmap**", fontsize=16)
                        st.pyplot(fig)
            else:
                st.warning("⚠️ Please select at least two numerical columns for multivariate visualization.")

def manage_columns(df):
    """
    Allows users to remove unwanted columns and manage existing ones.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: Updated dataset.
    """
    st.header("🗂️ **Manage Columns**")
    all_columns = df.columns.tolist()
    with st.expander("🔄 **Remove Columns**"):
        st.markdown("Select columns you wish to remove from the dataset.")
        with st.form("remove_columns_form"):
            columns_to_remove = st.multiselect(
                "🚮 **Select Columns to Remove**",
                all_columns,
                help="Select one or more columns to remove from the dataset."
            )
            submitted = st.form_submit_button("🗑️ **Remove Selected Columns**")
            if submitted:
                if columns_to_remove:
                    df = df.drop(columns=columns_to_remove)
                    st.success(f"✅ Removed columns: {', '.join(columns_to_remove)}")
                    st.write(f"**Updated Dataset Shape:** {df.shape}")
                    st.dataframe(df.head())
                else:
                    st.warning("⚠️ Please select at least one column to remove.")
    # Optionally, add more column management features here
    return df

def main():
    """
    The main function that orchestrates the Streamlit application.
    """
    st.set_page_config(
        page_title="📊 Outlier Detection and Analysis App",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📈 **Outlier Detection and Analysis App**")
    st.markdown("""
    Welcome to the **Outlier Detection and Analysis App**! This educational application allows you to:
    - 📥 **Upload** your dataset (CSV or Excel).
    - 🔍 **Filter** data based on custom conditions.
    - 🧮 **Aggregate** scores from multiple numerical columns and perform variable transformations.
    - 🗂️ **Manage** columns by removing unwanted ones.
    - 📊 **Visualize** data distributions (monovariate and multivariate).
    - 🔎 **Detect** outliers using various statistical methods.
    - 💾 **Download** the enhanced dataset with outlier flags.

    **Instructions:**
    1. **Upload** your dataset using the upload section.
    2. **Filter** the data by entering your conditions in the filter section.
    3. **Aggregate** scores and perform variable transformations if needed.
    4. **Manage** columns by removing any unwanted ones.
    5. **Visualize** the data distributions.
    6. **Choose** a statistical test to detect outliers.
    7. **Detect** outliers and review the results.
    8. **Download** the enhanced dataset with outlier information.

    **Educational Notes:**
    - **Outliers** are data points that deviate significantly from the majority of the data.
    - Detecting outliers is crucial as they can impact statistical analyses and machine learning models.
    - Different statistical methods are suitable for different data distributions and scenarios.
    - **Variable Transformation:** Techniques like log transformation or square root transformation can help in normalizing data or stabilizing variance.
    - **Aggregation:** Combining multiple numerical columns can create composite scores that summarize key aspects of the data.
    """)

    # Step 1: Upload Dataset
    df = upload_dataset()
    if df is not None:
        st.markdown("---")
        # Step 2: Filter Cases
        filtered_df = filter_cases(df)
        
        st.markdown("---")
        # Step 3: Aggregate Scores and Variable Transformation
        aggregated_df = aggregate_scores(filtered_df)
        
        st.markdown("---")
        # Step 4: Manage Columns
        cleaned_df = manage_columns(aggregated_df)
        
        st.markdown("---")
        # Step 5: Add Multiple Aggregated Scores & Variable Transformations
        cleaned_df = add_custom_variables(cleaned_df)
        
        st.markdown("---")
        # Step 6: Data Visualization
        st.header("📊 **Data Visualization**")
        col1, col2 = st.columns(2)
        with col1:
            plot_monovariate_distribution(cleaned_df)
        with col2:
            plot_multivariate_distribution(cleaned_df)
        
        st.markdown("---")
        # Step 7: Choose Statistical Test
        selected_test = choose_statistical_test()
        
        # Step 8: Identify Outliers
        outlier_df = identify_outliers(cleaned_df, selected_test)
        
        st.markdown("---")
        # Step 9: Download Enhanced Dataset
        download_enhanced_dataset(outlier_df)
        
        st.markdown("---")
        # Step 10: Display the enhanced dataset
        st.header("📂 **Enhanced Dataset**")
        st.dataframe(outlier_df)
        
        st.subheader("📊 **Dataset Statistics**")
        st.write(outlier_df.describe())
        
        # Step 11: Add Footer
        add_footer()

if __name__ == "__main__":
    main()
