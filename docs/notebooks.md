# 📓 NLP Course Notebooks

This directory contains all the Jupyter notebooks for the Advanced NLP Classes. These notebooks provide hands-on experience with the concepts covered in the lectures.

## ⚙️ Setup Instructions

### 🔧 Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)

### 📦 Installation

We use Poetry to manage dependencies. Follow these steps to set up your environment:

#### 1️⃣ Install Poetry

**macOS / Linux**:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows**:
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

#### 2️⃣ Clone the repository and install dependencies

```bash
git clone https://github.com/agombert/AdvancedNLPClasses.git
cd AdvancedNLPClasses
poetry install
```

#### 3️⃣ Install additional dependencies for notebooks

```bash
poetry add pandas numpy matplotlib scikit-learn spacy jupyter
poetry run python -m spacy download en_core_web_sm
```

#### 4️⃣ Launch Jupyter Notebook

```bash
poetry run jupyter notebook
```

Navigate to the `notebooks` directory to access all the notebooks.

### 🛠️ Troubleshooting

If you encounter issues with the installation:

- **macOS**: You might need to install Xcode command-line tools: `xcode-select --install`
- **Ubuntu**: Ensure you have build essentials: `sudo apt-get install build-essential`
- **Windows**: Make sure you have the Microsoft C++ Build Tools installed

## 📖 Table of Contents

### 🐍 Python Fundamentals (Session 1)

These notebooks cover the essential Python skills needed for NLP:

- **[Python Types](chapter1/Session_1_1_Python_1o1_1.ipynb)**: Understanding Python's type system, from basic to advanced types
- **[Python Classes](chapter1/Session_1_1_Python_1o1_2.ipynb)**: Object-oriented programming in Python
- **[Python Dataframes](chapter1/Session_1_1_Python_1o1_3.ipynb)**: Working with pandas for data manipulation
- **[Python NumPy](chapter1/Session_1_1_Python_1o1_4.ipynb)**: Numerical computing with NumPy
- **[Python scikit-learn](chapter1/Session_1_1_Python_1o1_5.ipynb)**: Introduction to machine learning with scikit-learn

### 📝 NLP Techniques (Session 1)

- **[Baseline with regexes and spaCy](chapter1/Session_1_2_baselines.ipynb)**: Implementing simple but effective baseline approaches
- **[TF-IDF: how to judge its quality?](chapter1/Session_1_3_tfidf.ipynb)**: Understanding and implementing TF-IDF
- **[BM25: a better TF-IDF, judge through different metrics](chapter1/Session_1_4_BM25.ipynb)**: Advanced information retrieval techniques

### 📝 Chapter 2: Neural Networks, Backpropagation & RNNs

- **[Intro to Neural Nets & Backprop (NumPy)](chapter2/Session_2_1_NeuralNets_with_Numpy.ipynb)**: Implementing neural networks with NumPy
- **[Simple RNN for Text Generation (Tiny Shakespeare)](chapter2/Session_2_2_Text_Generation_with_RNN.ipynb)**: Generating text with a simple RNN
- **[LSTM for Sequence Classification](chapter2/Session_2_3_LSTM_Classif.ipynb)**: Building an LSTM for sequence classification

### 📝 Chapter 3: Word Embeddings

- **[Word2Vec from Scratch - with negative sampling](chapter3/Session_3_1_Word2Vec_Training.ipynb)**: Implementing Word2Vec from scratch with negative sampling
- **[Embedding Evaluation: Intrinsic and Extrinsic](chapter3/Session_3_2_Embedding_Evaluation.ipynb)**: Evaluating word embeddings using both intrinsic and extrinsic metrics
- **[Classification with Embeddings](chapter3/Session_3_3_Embedding_Classification.ipynb)**: Using embeddings for classification tasks



## 🤝 Contributing

If you find errors or have suggestions for improving these notebooks, please open an issue or submit a pull request.

## 📄 License

These notebooks are provided for educational purposes as part of the Advanced NLP Classes at Barcelona School of Economics.
