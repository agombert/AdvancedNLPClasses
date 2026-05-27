## 🎓 Course Materials

### 📑 Slides

[Download Session 3 Slides (PDF)](../pdfs/BSE_NLP_Session_3.pdf)

[Download Bonus Slides — Interpreting Embeddings: Cosine vs. L2 (PDF)](../pdfs/BSE_NLP_Session_5_Embeddings.pdf)

### 📓 Notebooks

- [Word2Vec from Scratch - with negative sampling](Session_3_1_Word2Vec_Training.ipynb)
- [Embedding Evaluation: Intrinsic and Extrinsic](Session_3_2_Embedding_Evaluation.ipynb)
- [Classification with Embeddings](Session_3_3_Embedding_Classification.ipynb)
- [Interpreting Static Embeddings: Norm, Distance & Geometry](Session_3_4_Interpreting_Static_Embeddings.ipynb) — [bonus slides](../pdfs/BSE_NLP_Session_5_Embeddings.pdf)

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

- **Luhn, H. P. (1957): A Statistical Approach to Mechanized Encoding and Searching of Literary Information** – [Paper](https://ieeexplore.ieee.org/document/5392697)
  Early work on term-frequency statistics for indexing and search.

- **Spärck Jones, K. (1972): A Statistical Interpretation of Term Specificity and Its Application in Retrieval** – [Paper](https://www.emerald.com/insight/content/doi/10.1108/eb026526/full/html)
  Introduces inverse document frequency, the IDF in TF-IDF.

- **Manning, Raghavan & Schütze (2008): Introduction to Information Retrieval** – [Book](https://nlp.stanford.edu/IR-book/)
  Standard reference for vector-space retrieval, TF-IDF, and evaluation.

- **Mikolov et al. (2013): Distributed Representations of Words and Phrases and Their Compositionality** – [Paper](https://arxiv.org/abs/1310.4546)
  Companion Word2Vec paper introducing skip-gram with negative sampling.

- **McCann et al. (2017): Learned in Translation: Contextualized Word Vectors (CoVe)** – [Paper](https://arxiv.org/abs/1708.00107)
  Contextual word vectors derived from a machine-translation encoder.

- **Peters et al. (2017): Semi-supervised Sequence Tagging with Bidirectional Language Models** – [Paper](https://aclanthology.org/P17-1161/)
  Pre-LM-augmented sequence tagging, a precursor to ELMo.

- **Howard & Ruder (2018): Universal Language Model Fine-tuning for Text Classification (ULMFiT)** – [Paper](https://aclanthology.org/P18-1031/)
  Transfer-learning recipe for fine-tuning language models on downstream tasks.

- **Radford et al. (2019): Language Models are Unsupervised Multitask Learners (GPT-2)** – [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  Scaling autoregressive language models for zero-shot transfer.

#### Bonus — Interpreting Static Embeddings: Norm, Distance & Geometry

- **Schakel & Wilson (2015): Measuring Word Significance Using Distributed Representations of Words** – [Paper](https://arxiv.org/abs/1508.02297)
  Shows embedding norm grows with word frequency and significance.

- **Mu & Viswanath (2018): All-but-the-Top: Simple and Effective Postprocessing for Word Representations** – [Paper](https://arxiv.org/abs/1702.01417)
  Removing the top principal components (mostly frequency) improves embeddings.

- **Ethayarajh (2019): How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings** – [Paper](https://arxiv.org/abs/1909.00512)
  Documents anisotropy: contextual embeddings occupy a narrow cone.

- **Timkey & van Schijndel (2021): All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality** – [Paper](https://arxiv.org/abs/2109.04404)
  A few rogue dimensions dominate cosine similarity; standardization fixes it.

- **Su et al. (2021): Whitening Sentence Representations for Better Semantics and Faster Retrieval** – [Paper](https://arxiv.org/abs/2103.15316)
  Whitening removes anisotropy and reduces embedding dimensionality.

- **Reimers & Gurevych (2019): Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** – [Paper](https://arxiv.org/abs/1908.10084)
  Siamese fine-tuning to produce semantically meaningful sentence embeddings.

- **Gao et al. (2021): SimCSE: Simple Contrastive Learning of Sentence Embeddings** – [Paper](https://arxiv.org/abs/2104.08821)
  Contrastive objective that yields isotropic, high-quality sentence embeddings.

- **Beyer et al. (1999): When Is "Nearest Neighbor" Meaningful?** – [Paper](https://link.springer.com/chapter/10.1007/3-540-49257-7_15)
  Curse of dimensionality: distance contrast vanishes in high dimensions.

---

### 💻 Practical Components

- **From Scratch Word2Vec**: We walk through how Skip-Gram is trained using pairs of target/context words and how to integrate negative sampling.
- **Embedding Visualizations**: Use t-SNE or PCA to project high-dimensional embeddings and see how similar words cluster.
- **Text Classification with Embeddings**: Test embeddings in real classification tasks with logistic regression or LSTMs.
- **Using Pretrained ELMo Embeddings**: Fine-tune contextual embeddings on your own dataset.
