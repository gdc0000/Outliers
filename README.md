# Outlier Detection and Analysis App

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://outlier-detection.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

The **Outlier Detection and Analysis App** is an interactive, educational tool built with Streamlit for data analysts and researchers. This application allows you to:

- **Upload** your dataset (CSV or Excel) and preview its contents.
- **Filter** records using custom conditions with pandas query syntax.
- **Aggregate** scores and perform variable transformations on numerical columns.
- **Manage** and remove unwanted columns from your dataset.
- **Visualize** data distributions with a variety of plots (monovariate and multivariate).
- **Detect Outliers** using multiple statistical methods such as Z-Score, IQR, DBSCAN, and Isolation Forest.
- **Download** the enhanced dataset with outlier flags for further analysis.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

The Outlier Detection and Analysis App is designed to help you explore, transform, and clean your data while detecting potential outliers. With a step-by-step workflow that includes data filtering, aggregation, transformation, visualization, and outlier detection, this application is ideal for:

- Educational purposes and data exploration.
- Preprocessing data for statistical analysis or machine learning.
- Understanding the impact of outliers on your dataset.

---

## Features

- **Dataset Upload and Preview:**  
  Upload CSV or Excel files via the sidebar and view a preview of your data.

- **Case Filtering:**  
  Apply custom filter conditions using pandas query syntax to subset your data.

- **Score Aggregation & Variable Transformation:**  
  Combine multiple numerical columns into composite scores using methods such as sum, mean, or weighted sum. Apply transformations like log or square root (or even custom expressions) to normalize or stabilize variance.

- **Column Management:**  
  Remove unwanted columns and add custom variables to further enrich your dataset.

- **Data Visualization:**  
  Generate monovariate plots (Histogram with KDE, Box Plot, Violin Plot) and multivariate visualizations (Pair Plots, Scatter Plots, Correlation Heatmaps) to explore data distributions.

- **Outlier Detection:**  
  Detect outliers in selected numerical columns using a choice of statistical tests:
  - **Z-Score:** Identify data points beyond a specified standard deviation threshold.
  - **IQR:** Use the interquartile range to flag outliers.
  - **DBSCAN:** Cluster data points and mark those that do not belong to any cluster.
  - **Isolation Forest:** Leverage an ensemble method to isolate anomalies in high-dimensional data.

- **Download Enhanced Dataset:**  
  Download your modified dataset (with outlier flags) as an Excel file.

- **Educational Insights:**  
  Detailed instructions and notes are provided within the app to help you understand each step and the methods used.

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/outlier-detection-analysis-app.git
   cd outlier-detection-analysis-app
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the Required Dependencies**

   All required packages are listed in the [`requirements.txt`](./requirements.txt) file. Install them using pip:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Run the Streamlit Application**

   Launch the app with the following command:

   ```bash
   streamlit run main.py
   ```

2. **Follow the Step-by-Step Workflow**

   - **Upload Dataset:**  
     Use the sidebar to upload a CSV or Excel file and preview the dataset.

   - **Filter Cases:**  
     Enter your custom filter conditions (e.g., `AGE > 30 and INCOME < 50000`) to subset the data.

   - **Aggregate Scores & Transform Variables:**  
     Select numerical columns to aggregate using Sum, Mean, or Weighted Sum, and apply variable transformations like log or square root.

   - **Manage Columns:**  
     Remove any unwanted columns to clean your dataset.

   - **Add Custom Operations:**  
     Add multiple custom aggregated scores or transformations with your own custom names.

   - **Visualize Data:**  
     Generate monovariate and multivariate plots to inspect the distributions of your numerical variables.

   - **Outlier Detection:**  
     Choose a statistical test (Z-Score, IQR, DBSCAN, Isolation Forest) to detect outliers in the selected columns and review the outlier records.

   - **Download Dataset:**  
     Download the enhanced dataset with outlier flags as an Excel file.

---

## File Structure

```
.
├── main.py           # Main Streamlit application code for outlier detection and analysis
├── requirements.txt  # List of required Python packages
└── README.md         # This file
```

---

## Requirements

The app requires the following packages:

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- xlsxwriter

For full version requirements, please refer to the [`requirements.txt`](./requirements.txt) file.

---

## Contributing

Contributions are welcome! If you have suggestions, bug fixes, or improvements:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Open a pull request with a detailed description of your modifications.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Contact

**Gabriele Di Cicco, PhD in Social Psychology**  
[GitHub](https://github.com/gdc0000) | [ORCID](https://orcid.org/0000-0002-1439-5790) | [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)

---

Happy Analyzing!
