# Python NumPy

NumPy is the fundamental package for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

[Raw Notebook](https://github.com/agombert/AdvancedNLPClasses/blob/main/notebooks/support/Session_1_1_Python_1o1_4.ipynb)

## Table of Contents
1. [Basic Concepts](#basic)
2. [Advanced Concepts](#advanced)
3. [Exercises](#exercises)
4. [Real-World Applications](#applications)

---

## 1. Basic Concepts <a name="basic"></a>

### 1.1 NumPy Arrays

```python
import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([[1, 2], [3, 4]])

print("arr1:", arr1)
print("arr2:\n", arr2)
print("Shape of arr2:", arr2.shape)  # Display the shape of arr2
```

### 1.2 Array Operations

NumPy allows element-wise operations for arithmetic, comparisons, etc.

```python
arr3 = np.array([10, 20, 30])

# Element-wise arithmetic (adds corresponding elements of arr1 and arr3)
print("arr1 + arr3 =", arr1 + arr3)

# Scalar operations (multiplies each element of arr1 by 2)
print("arr1 * 2 =", arr1 * 2)

# Comparison (checks if elements in arr3 are greater than 15)
print("arr3 > 15?", arr3 > 15)
```

### 1.3 Slicing and Indexing

```python
# Create a 1D array
arr4 = np.array([10, 11, 12, 13, 14, 15])
print("arr4[1:4] =", arr4[1:4])  # Slicing: elements from index 1 to 3

# Create a 2D array
arr5 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("\n2D array:\n", arr5)
print("arr5[0, 1] =", arr5[0, 1])  # Access element at row 0, column 1
print("arr5[:, 1] =", arr5[:, 1])  # Access all rows in column 1
```

---

## 2. Advanced Concepts <a name="advanced"></a>

### 2.1 Broadcasting

NumPy can automatically expand array dimensions to match shapes during operations.

```python
arr6 = np.array([[1, 2, 3], [4, 5, 6]])
arr7 = np.array([10, 20, 30])
# arr6 is (2,3) and arr7 is (3,)
# Broadcasting: arr7 is added to each row of arr6
result = arr6 + arr7
print(result)
```

### 2.2 Reshaping and Transposing

```python
arr8 = np.arange(1, 7)  # Create a 1D array with numbers 1 to 6
print("Original:", arr8)

arr8_reshaped = arr8.reshape(2, 3)  # Reshape into a 2x3 array
print("\nReshaped to (2,3):\n", arr8_reshaped)

# Transpose the reshaped array
print("\nTransposed:\n", arr8_reshaped.T)
```

### 2.3 Mathematical and Statistical Functions

NumPy provides a variety of built-in functions for computations like sum, mean, std, etc.

```python
arr9 = np.array([1, 2, 3, 4, 5])
print("Sum:", np.sum(arr9))
print("Mean:", np.mean(arr9))
print("Standard Deviation:", np.std(arr9))
```

### 2.4 Random Module

Useful for generating random numbers, random samples, etc.

```python
# Generate a 3x3 array of random numbers from a uniform distribution between 0 and 1
random_arr = np.random.rand(3, 3)
print("Random Array:\n", random_arr)

# Generate a 2x5 array of random integers between 0 and 10
randint_arr = np.random.randint(0, 10, size=(2, 5))
print("\nRandom Integers:\n", randint_arr)
```

---

## 3. Exercises <a name="exercises"></a>

### Exercise 1: Basic Array Operations
1. Create a NumPy array of shape `(4,4)` with numbers from 1 to 16.
2. Print the slice containing the second row.
3. Multiply the entire array by 2.

```python
# Your code here
```

### Exercise 2: Reshaping and Broadcasting
1. Create a 1D array with numbers from 1 to 9.
2. Reshape it into `(3,3)`.
3. Add a 1D array `[10,10,10]` to it using broadcasting.

```python
# Your code here
```

### Exercise 3: Statistical Functions
1. Generate an array of 100 random numbers using `np.random.randn()`.
2. Compute the mean and standard deviation.
3. Print the values.

```python
# Your code here
```

---

## 4. Real-World Applications <a name="applications"></a>

### Machine Learning
- NumPy arrays are the foundation for libraries like scikit-learn, TensorFlow, and PyTorch.

### Linear Algebra
- Operations like matrix multiplication, decompositions, etc. are common in ML and data analysis.

### Signal Processing, Simulations
- Researchers often use NumPy for large-scale numerical simulations, random processes, or DSP tasks.

NumPy is indispensable for scientific computing in Python, serving as the backbone for almost every advanced data library!
