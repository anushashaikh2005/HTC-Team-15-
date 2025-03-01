Advanced Data Cleaner — Enterprise-Grade CSV Data Cleaning System
Welcome to Advanced Data Cleaner, a comprehensive Python tool for cleaning, profiling, and validating CSV datasets. This system is built to handle large-scale, enterprise-grade data transformations with configurable strategies for missing values, outliers, data types, duplicates, normalization, and more.

Key Features
Automatic CSV Ingestion

Auto-detects file encoding and delimiter (or configure them manually).
Reads large files in chunks for improved performance.
Configurable Missing Value Handling

Supports dropping rows/columns, multiple imputation methods (mean, median, KNN, iterative), or flagging missing data with indicator columns.
Robust Outlier Detection

Offers Z-score, IQR, Isolation Forest, or no outlier detection.
Choose to remove, cap, or flag outliers automatically or based on thresholds.
Powerful Data Type Management

Detects and converts strings to dates, booleans, categories, or numerics automatically.
Flexible manual type mappings where needed.
Flexible Duplicate Handling

Removes or flags duplicates (keep first, keep last, remove all).
Normalization & Encoding

Standardize, min-max, robust, or log transformations for numeric data.
Auto-encodes categorical data using label, one-hot, frequency, or target encoding.
Data Validation & Rules

Define custom rules (range checks, regex, required fields) and specify actions (flag, drop, fix).
Performance & Scaling

Parallel processing for large datasets.
Optimizes DataFrame memory usage.
Detailed Reporting

Generates HTML or JSON profile reports for both original and cleaned data.
Comparison report highlights changes in rows, columns, memory usage, and distributions.
Basic Usage
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
(Make sure the requirements file includes necessary libraries like pandas, numpy, scikit-learn, PyYAML, etc.)

Run via Command Line

bash
Copy
Edit
python advanced_data_cleaner.py --input your_data.csv --output cleaned_data.csv
This reads your_data.csv, cleans it using default/auto settings, and writes cleaned_data.csv.

Specify Configuration File
For advanced scenarios, create or generate a YAML/JSON config file:

bash
Copy
Edit
python advanced_data_cleaner.py --generate-config default_config.yaml
Then edit and use that file:

bash
Copy
Edit
python advanced_data_cleaner.py --input your_data.csv --config default_config.yaml
Adjust Pipeline Steps
You can choose which pipeline steps to run or skip:

bash
Copy
Edit
python advanced_data_cleaner.py --input your_data.csv \
    --pipeline read_data,detect_types,handle_missing,write_data \
    --skip-steps handle_outliers,encode_categories
Filtering & Parameters

Keep or drop specific columns:
bash
Copy
Edit
--keep-columns "col1,col2,col3"
--drop-columns "col4"
Outlier detection method:
bash
Copy
Edit
--outlier-method iqr --outlier-action cap
Missing data strategy:
bash
Copy
Edit
--missing-strategy impute --missing-threshold-row 0.6
