# Python scikit-learn

scikit-learn is a popular machine learning library offering a wide range of tools for predictive modeling: classification, regression, clustering, and more.

[Raw Notebook](https://github.com/agombert/AdvancedNLPClasses/blob/main/notebooks/support/Session_1_1_Python_1o1_5.ipynb)

## Table of Contents
1. [Basic Concepts](#basic)
2. [Advanced Concepts](#advanced)
3. [Exercises](#exercises)
4. [Real-World Applications](#applications)

---

## 1. Basic Concepts <a name="basic"></a>

### 1.1 Classifiers

Most scikit-learn APIs follow the `fit` and `predict` pattern.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Example dataset (XOR-like)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict on the same data (for demonstration)
preds = model.predict(X)
print("Predictions:", preds)
print("Accuracy:", accuracy_score(y, preds))
```

### 1.2 Regression

```python
from sklearn.linear_model import LinearRegression

# Simple regression example with one feature per sample
X_reg = np.array([[1], [2], [3], [4], [5]])  # Features
y_reg = np.array([2, 4, 5, 4, 5])  # Targets

# Create and train the linear regression model
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

# Print the model's learned parameters
print("Coefficients:", reg_model.coef_)
print("Intercept:", reg_model.intercept_)

# Predict using the model
y_pred = reg_model.predict(X_reg)
print("Predictions:", y_pred)
```

---

## 2. Advanced Concepts <a name="advanced"></a>

### 2.1 Pipelines

A pipeline chains multiple transformations and a final estimator into one unified model.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Create a pipeline with a StandardScaler and a Support Vector Classifier
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Scales features to have zero mean and unit variance
    ("svc", SVC(kernel="linear"))    # SVC with a linear kernel
])

# Synthetic data for demonstration
X2 = np.array([[1, 200], [2, 180], [3, 240], [4, 210]])
y2 = np.array([0, 0, 1, 1])

# Fit the pipeline on the data and predict
pipeline.fit(X2, y2)
pred2 = pipeline.predict(X2)
print("Pipeline predictions:", pred2)
```

### 2.2 Model Selection and Cross-Validation

scikit-learn provides utilities like `GridSearchCV` for hyperparameter tuning and cross-validation.

```python
from sklearn.model_selection import GridSearchCV

# Define a parameter grid for the SVC within the pipeline
param_grid = {
    "svc__C": [0.1, 1, 10],
    "svc__kernel": ["linear", "rbf"]
}

# Create a GridSearchCV object using the pipeline and the parameter grid, with 2-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=2)
grid_search.fit(X2, y2)
print("Best Params:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

### 2.3 Feature Engineering

Feature engineering transforms raw data into features suitable for model training. scikit-learn includes tools like `PolynomialFeatures` and text vectorizers (e.g., `CountVectorizer`).

```python
from sklearn.preprocessing import PolynomialFeatures

# Example data: a 1D array of numbers
X_poly = np.array([[2], [3], [4]])
poly = PolynomialFeatures(degree=2)
X_transformed = poly.fit_transform(X_poly)

print("Original:", X_poly)
print("Polynomial Features:\n", X_transformed)
```

### 2.4 Common Metrics

In addition to accuracy, scikit-learn offers metrics such as `precision_score`, `recall_score`, `f1_score`, `r2_score`, etc.

```python
from sklearn.metrics import precision_score, recall_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]

print("Precision:", precision_score(y_true, y_pred))
print("Recall:   ", recall_score(y_true, y_pred))
```

---

## 3. Exercises <a name="exercises"></a>

### Exercise 1: Classification
1. Use any small dataset (or generate synthetic data) for a classification task.
2. Train a logistic regression model.
3. Print accuracy.

```python
# Your code here
X_test = np.array([[...], [...], ...])
y_test = np.array([...])
# logistic_model = LogisticRegression()
# logistic_model.fit(X_test, y_test)
# preds_test = logistic_model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, preds_test))
```

### Exercise 2: Pipeline
1. Create a pipeline with a `StandardScaler` and a `KNeighborsClassifier`.
2. Fit it on some toy dataset.
3. Predict and evaluate the performance.

```python
# Your code here
```

### Exercise 3: Grid Search
1. Use `GridSearchCV` on any model of your choice (e.g., `SVC`).
2. Print the best parameters and best score.

```python
# Your code here
```

---

## 4. Real-World Applications <a name="applications"></a>

### Classification Tasks
- **Spam Detection**: Email text classification.
- **Image Recognition**: Digits dataset or complex images.

### Regression Tasks
- **House Price Prediction**: Predicting real estate prices based on features.
- **Stock Forecasting**: Although more advanced time-series methods exist, scikit-learn can handle simple regression or feature-based approaches.

### Clustering
- **Customer Segmentation**: Using KMeans or DBSCAN to group similar customers.

### Model Deployment
- scikit-learn models can be saved (e.g., using `joblib`) and deployed within web applications for real-time inference.

scikit-learn’s consistent API and wide range of algorithms make it a go-to toolkit for ML in Python.
