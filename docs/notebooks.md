# NLP Course Notebooks

This directory contains all the Jupyter notebooks for the Advanced NLP Classes. These notebooks provide hands-on experience with the concepts covered in the lectures.

## Table of Contents

### Python Fundamentals (Session 1)

These notebooks cover the essential Python skills needed for NLP:

- **[Python Types](../notebooks/support/Session_1_1_Python_1o1_1.ipynb)**: Understanding Python's type system, from basic to advanced types
- **[Python Classes](../notebooks/support/Session_1_1_Python_1o1_2.ipynb)**: Object-oriented programming in Python
- **[Python Dataframes](../notebooks/support/Session_1_1_Python_1o1_3.ipynb)**: Working with pandas for data manipulation
- **[Python NumPy](../notebooks/support/Session_1_1_Python_1o1_4.ipynb)**: Numerical computing with NumPy
- **[Python scikit-learn](../notebooks/support/Session_1_1_Python_1o1_5.ipynb)**: Introduction to machine learning with scikit-learn

### NLP Techniques (Session 1)

- **[Baseline with regexes and spaCy](../notebooks/support/Session_1_2_baselines.ipynb)**: Implementing simple but effective baseline approaches
- **[TF-IDF: how to judge its quality?](../notebooks/support/Session_1_3_TFIDF.ipynb)**: Understanding and implementing TF-IDF
- **[BM25: a better TF-IDF, judge through different metrics](../notebooks/support/Session_1_4_BM25.ipynb)**: Advanced information retrieval techniques

## Learning Objectives

Each notebook is designed with specific learning objectives:

1. **Python Fundamentals**: Ensure you have the necessary programming skills for NLP
2. **Baseline Approaches**: Learn to implement and evaluate simple NLP solutions
3. **Text Representation**: Understand how to convert text into numerical representations
4. **Evaluation Metrics**: Learn how to properly evaluate NLP models

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)

### Installation

We use Poetry to manage dependencies. Follow these steps to set up your environment:

#### 1. Install Poetry

**macOS / Linux**:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows**:
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

#### 2. Clone the repository and install dependencies

```bash
git clone https://github.com/agombert/AdvancedNLPClasses.git
cd AdvancedNLPClasses
poetry install
```

#### 3. Install additional dependencies for notebooks

```bash
poetry add pandas numpy matplotlib scikit-learn spacy jupyter
poetry run python -m spacy download en_core_web_sm
```

#### 4. Launch Jupyter Notebook

```bash
poetry run jupyter notebook
```

Navigate to the `notebooks` directory to access all the notebooks.

### Troubleshooting

If you encounter issues with the installation:

- **macOS**: You might need to install Xcode command-line tools: `xcode-select --install`
- **Ubuntu**: Ensure you have build essentials: `sudo apt-get install build-essential`
- **Windows**: Make sure you have the Microsoft C++ Build Tools installed

## Contributing

If you find errors or have suggestions for improving these notebooks, please open an issue or submit a pull request.

## License

These notebooks are provided for educational purposes as part of the Advanced NLP Classes at Barcelona School of Economics.
