## 🎓 Course Materials - Practical NLP - 1

## Session 4: Text Classification Pipelines and Explainability

In this hands-on session, we walk through the evolution of **text classification pipelines**, from traditional approaches like **TF-IDF + linear classifiers** to modern **deep learning models with LSTM** and **pretrained word embeddings** like **Word2Vec**. The session closes with an introduction to **model explainability** using **LIME**, giving students insight into how models make decisions.

This notebook is designed as a modular blueprint that can be reused and extended for many text classification tasks.

### 📓 Notebooks

- [Session 4: From TF-IDF to LSTMs and Explainability](Session_4.ipynb)
- [Session 4: correction]()

---

### 🎯 Learning Objectives

1. Understand how to turn raw text into machine-readable input (TF-IDF, tokenization, embeddings).
2. Build baseline and deep models (logistic regression, BiLSTM).
3. Integrate **pre-trained embeddings** (Word2Vec) into custom pipelines.
4. Apply **explainability tools** like LIME to interpret model behavior.
5. Compare models using quantitative and qualitative evaluation (metrics & examples).

---

### 📖 Bibliography & Recommended Reading

- **scikit-learn TF-IDF Documentation** – [Link](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- **spaCy Tokenizer Docs** – [Link](https://spacy.io/api/tokenizer)
- **Gensim Word2Vec Tutorial** – [Link](https://radimrehurek.com/gensim/models/word2vec.html)
- **LIME GitHub Repo** – [Link](https://github.com/marcotcr/lime)

---

### 💻 Practical Components

- 🧪 Build a **text classifier from scratch** using `TF-IDF + LogisticRegression`.
- 🧠 Train an **LSTM** with one-hot or pre-trained embeddings.
- 📦 Use **Word2Vec** embeddings from Hugging Face.
- 🔍 Explain and debug predictions with **LIME** for real-world NLP workflows.
- 🎯 Compare models using both **metrics** and **example-level outputs**.
