# Advanced NLP Classes

This repository contains materials for the Advanced Natural Language Processing course taught at Barcelona School of Economics.

## Course Overview

This course navigates the evolution of Natural Language Processing (NLP) from foundational techniques to advanced concepts like Large Language Models and ChatGPT. It begins with core principles such as TF-IDF and word embeddings, advancing through deep learning innovations like LSTM and BERT.

The course is structured into three main parts:
1. Good old fashioned NLP (Sessions 1-4)
2. Almost part of good old fashioned NLP (Sessions 5-8)
3. LLMs, Agents & Others (Sessions 9 & 10)

## Repository Structure

- **docs/**: Course documentation and lecture notes
  - **pdfs/**: PDF versions of lecture slides
  - **chapter\*/**: Content for each session
- **notebooks/**: Jupyter notebooks for hands-on exercises

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/agombert/AdvancedNLPClasses.git
cd AdvancedNLPClasses
```

2. **Install dependencies with Poetry**

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

3. **Set up notebook environment**

```bash
# Download necessary models and datasets
poetry run setup-notebooks
```

4. **Start the documentation server**

```bash
poetry run mkdocs serve
```

5. **Launch Jupyter for notebooks**

```bash
poetry run jupyter notebook
```

## Course Materials

- **Documentation**: Visit the [course website](https://agombert.github.io/AdvancedNLPClasses/) for comprehensive materials
- **Notebooks**: Explore the [notebooks directory](notebooks/) for hands-on exercises
- **Slides**: Download lecture slides from the [pdfs directory](docs/pdfs/)

## Contributing

If you find errors or have suggestions for improving the course materials, please open an issue or submit a pull request.
