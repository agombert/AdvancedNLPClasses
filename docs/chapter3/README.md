## 🎓 Course Materials

### 📑 Slides

[Download Session 3 Slides (PDF)](../pdfs/2025_BSE_NLP_Session_3.pdf)

### 📓 Notebooks

- [Word2Vec from Scratch - with negative sampling](Session_3_1_Word2Vec_Training.ipynb)
- [Embedding Evaluation: Intrinsic and Extrinsic](Session_3_2_Embedding_Evaluation.ipynb)
- [Contextualized Embeddings with ELMo vs Static Embeddings](Session_3_3_Contextual_Embeddings.ipynb)

---

## 🚀 Session 3: Word Embeddings

In this third session, we explore how words can be **mathematically represented** and why this is essential in any NLP pipeline. We trace the journey from traditional **sparse one-hot encodings** and **TF-IDF** vectors to powerful **dense embeddings** like **Word2Vec** and **GloVe**, and finally to **context-aware models** like **ELMo** and **BERT**.

We also see how these embeddings are **evaluated** and how they can be applied to downstream NLP tasks like **sentiment analysis**, **NER**, or **question answering**.

### 🎯 Learning Objectives

1. Understand the **limitations of traditional word representations** (e.g., sparsity, context insensitivity).
2. Learn how **dense vector embeddings** solve these problems and how to train them.
3. Explore **Word2Vec architectures** (Skip-gram and CBOW) and techniques like **negative sampling**.
4. Evaluate embeddings both **intrinsically** (e.g., word similarity, analogy) and **extrinsically** (e.g., classification).
5. Discover the next evolution: **contextual embeddings** with **ELMo**, including how to **pretrain** and **fine-tune** them.

---

### 📚 Topics Covered

#### Static Word Embeddings

- One-hot, TF-IDF: Why we moved beyond them.
- **Word2Vec** (Skip-gram, CBOW) and the training process.
- **Negative Sampling**: How to make training efficient.
- **GloVe**: A count-based alternative to Word2Vec.
- **FastText**: Subword-level embeddings to deal with rare words and misspellings.

#### Evaluating Word Embeddings

- **Intrinsic evaluations**:
  - Word similarity (e.g., cosine distance between “king” and “queen”).
  - Word analogy (“man” : “woman” :: “king” : “queen”).
- **Extrinsic evaluations**:
  - How well embeddings help in downstream tasks like classification or POS tagging.

#### Contextual Word Embeddings

- Why static vectors fall short (e.g., "bank" in “river bank” vs. “bank account”).
- Introduction to **ELMo** (Peters et al., 2018).
- **Bidirectional Language Modeling** using LSTMs.
- How ELMo generates different embeddings for the same word in different contexts.
- Using ELMo for **transfer learning** in real-world NLP tasks (e.g., sentiment classification).

---

### 🧠 Key Takeaways

| Aspect                     | Static Embeddings              | Contextual Embeddings        |
|---------------------------|-------------------------------|------------------------------|
| Meaning Based on Context? | ❌ Same vector regardless      | ✅ Different vectors per use |
| Polysemy Handling         | ❌ No                         | ✅ Yes                       |
| Requires Large Corpus?    | ✅ Usually                    | ✅ Definitely                |
| Adaptable to Tasks?       | ⚠️ Not easily                 | ✅ Via fine-tuning           |

---

### 📖 Bibliography & Recommended Reading

- **Jay Alammar (2017): Visual Introduction to Word Embeddings** – [Blog Post](https://jalammar.github.io/illustrated-word2vec/)
  Excellent visuals to understand Word2Vec and GloVe.

- **Sebastian Ruder (2017): On Word Embeddings – Part 2: Approximating Co-occurrence Matrices** – [Blog Post](http://ruder.io/word-embeddings-2017/)
  Detailed breakdown of how different embedding models compare.

- **Mikolov et al. (2013): Efficient Estimation of Word Representations in Vector Space** – [Paper](https://arxiv.org/abs/1301.3781)
  The original Word2Vec paper introducing Skip-gram and CBOW models.

- **Pennington et al. (2014): GloVe: Global Vectors for Word Representation** – [Paper](https://nlp.stanford.edu/pubs/glove.pdf)
  Count-based embedding approach from Stanford NLP group.

- **Joulin et al. (2016): Bag of Tricks for Efficient Text Classification (FastText)** – [Paper](https://arxiv.org/abs/1607.01759)
  A very practical take on embeddings using subword units.

- **Peters et al. (2018): Deep Contextualized Word Representations** – [Paper](https://arxiv.org/abs/1802.05365)
  ELMo paper showing how dynamic embeddings outperform static ones on many tasks.

---

### 💻 Practical Components

- **From Scratch Word2Vec**: We walk through how Skip-Gram is trained using pairs of target/context words and how to integrate negative sampling.
- **Embedding Visualizations**: Use t-SNE or PCA to project high-dimensional embeddings and see how similar words cluster.
- **Text Classification with Embeddings**: Test embeddings in real classification tasks with logistic regression or LSTMs.
- **Using Pretrained ELMo Embeddings**: Fine-tune contextual embeddings on your own dataset.
