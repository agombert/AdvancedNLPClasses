# Python DataFrames (pandas)

This notebook introduces the basics and advanced features of pandas `DataFrame`. DataFrames are central to data manipulation in Python.

[Raw Notebook](https://github.com/agombert/AdvancedNLPClasses/blob/main/notebooks/support/Session_1_1_Python_1o1_3.ipynb)

## Table of Contents
1. [Basic Concepts](#basic)
2. [Advanced Concepts](#advanced)
3. [Exercises](#exercises)
4. [Real-World Applications](#applications)

---

## 1. Basic Concepts <a name="basic"></a>

### 1.1 Creating a DataFrame

```python
import pandas as pd

# From a dictionary:
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)
df  # Display the DataFrame
```

### 1.2 Reading Data from Files

Commonly, pandas is used to read CSV, Excel, or JSON files.

```python
# Example (uncomment if you have a file)
# df_csv = pd.read_csv('data.csv')
# df_excel = pd.read_excel('data.xlsx')
# df_json = pd.read_json('data.json')
pass  # Placeholder since no file is provided
```

### 1.3 Basic Inspection

Methods for quickly assessing your DataFrame’s shape and contents.

```python
# Display the first few rows
print(df.head())

# Display the last few rows
print(df.tail())

# Show summary information (data types, non-null counts)
print(df.info())

# Show statistical summary (only for numerical columns)
print(df.describe())
```

---

## 2. Advanced Concepts <a name="advanced"></a>

### 2.1 Indexing and Selection

Pandas offers powerful indexing with `.loc` (label-based) and `.iloc` (integer-based).

```python
# Label-based indexing: select the "Name" of the first row (index 0)
print("Label-based indexing:")
print(df.loc[0, "Name"])
print()

# Integer-based indexing: select the element at row index 1 and column index 2 (City)
print("Integer-based indexing:")
print(df.iloc[1, 2])
```

### 2.2 Merging and Joining

You can combine DataFrames in various ways using `merge()`, `join()`, or `concat()`.

```python
# Create an extra DataFrame with additional data
data_extra = {
    "Name": ["Alice", "Bob"],
    "Salary": [70000, 80000]
}
df_extra = pd.DataFrame(data_extra)

# Perform a left merge on the "Name" column
merged_df = pd.merge(df, df_extra, on="Name", how="left")
merged_df  # Display the merged DataFrame
```

### 2.3 GroupBy and Aggregation

Grouping data by categories and applying aggregate functions like `sum`, `mean`, or `count`.

```python
# Example sales data
df_sales = pd.DataFrame({
    "Product": ["A", "A", "B", "B", "B"],
    "Sales": [100, 150, 200, 120, 180],
    "Region": ["North", "South", "North", "South", "North"]
})

# Group by the "Product" column and calculate total sales for each product
grouped = df_sales.groupby("Product").agg({"Sales": "sum"})
grouped  # Display the aggregated results
```

### 2.4 Handling Missing Data

Missing data is common in real datasets. Pandas provides methods like `dropna()`, `fillna()`, etc.

```python
# Create a DataFrame with missing values
df_missing = pd.DataFrame({
    "Col1": [1, None, 3],
    "Col2": [None, 5, 6]
})
print(df_missing)

# Drop rows with any missing values
df_dropped = df_missing.dropna()
print("\nAfter dropna:\n", df_dropped)

# Fill missing values with 0
df_filled = df_missing.fillna(0)
print("\nAfter fillna(0):\n", df_filled)
```

---

## 3. Exercises <a name="exercises"></a>

### Exercise 1: Data Cleaning
1. Create a DataFrame with columns `Name`, `Age`, `City`, and some missing values.
2. Drop rows with missing values.
3. Fill missing values in `Age` with the mean age.

```python
# Your code here
import numpy as np

df_ex = pd.DataFrame({
    "Name": ["Tom", "Jane", "Steve", "NaN"],
    "Age": [25, None, 30, 22],
    "City": ["Boston", "", "Seattle", None]
})
# 1) Create the DataFrame
# 2) Drop rows with missing values
# 3) Fill missing Age with mean
```

### Exercise 2: GroupBy and Aggregation
Using the `df_sales` DataFrame shown earlier (or create your own):
1. Group by `Region`.
2. Calculate the average sales per region.
3. Print the results.

```python
# Your code here
```

### Exercise 3: Merging DataFrames
1. Create two DataFrames `df1` and `df2` with a common column (e.g., `id`).
2. Perform a left merge on `id`.
3. Perform an inner merge on `id`.

```python
# Your code here
```

---

## 4. Real-World Applications <a name="applications"></a>

### ETL (Extract, Transform, Load)
- Data scientists use pandas to extract data from various sources (databases, APIs, files), transform it (cleaning, feature engineering), and load it into analytics tools.

### Exploratory Data Analysis (EDA)
- Pandas is essential for quick EDA: summarizing datasets, detecting outliers, etc.

### Time-Series Analysis
- Pandas offers specialized support for time-series data, making it popular in finance and IoT data processing.

These are just a few examples—pandas is central to nearly every data-related task in Python!
