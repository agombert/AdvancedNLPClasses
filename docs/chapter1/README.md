
## Course Materials

### Slides:

[Download Session 1 Slides (PDF)](../pdfs/2025_BSE_NLP_Session_1.pdf)

### Notebooks:

- [Baseline with regexes](../notebooks/2025_BSE_NLP_Session_1.zip)
- [TF-IDF: how to judge its quality?](../notebooks/2025_BSE_NLP_Session_1.zip)
- [BM25: a better TF-IDF, judge through different metrics](../notebooks/2025_BSE_NLP_Session_1.zip)

---

## Session 1: Baselines and Sparse Representations

In our first session, I’ll introduce you to **baseline approaches**—simple yet powerful starting points for many NLP tasks. These baselines serve as reference points, helping you measure whether your more sophisticated models actually bring improvements. We’ll also explore the concept of **sparse representations**, such as bag-of-words or TF-IDF, which have been fundamental in text analysis for years.

### Learning Objectives

1. Grasp the **challenges** in processing natural language data.
2. Understand and create **sparse vector** representations of text.
3. Explore **baseline models** for typical NLP tasks.
4. Learn about **model evaluation** and choosing the right metrics.
5. Gain **hands-on practice** by building basic NLP pipelines.

Even if you have little background in NLP, don’t worry. This session walks you through each concept step by step and shows you how to practically implement them in Python.

### Topics Covered

#### Baseline & Evaluations

We’ll discuss how to set up clear baselines for NLP tasks—starting with basic data cleaning, tokenization, and simple modeling like bag-of-words classifiers. This section underlines why **evaluating** your model using metrics such as accuracy, F1-score, or BLEU is crucial in understanding how well the model performs and whether advanced methods are truly an improvement.

#### TF-IDF and Improvements

I’ll walk you through the **TF-IDF (Term Frequency–Inverse Document Frequency)** technique, explaining why it’s such a popular step beyond a raw bag-of-words. From there, we’ll dive into tweaks and improvements, including dimensionality reduction techniques, vector space models, and uses in information retrieval and text classification.

### Recommended Reading

- **Van Rijsbergen, C. J. (1979). _Information Retrieval (2nd ed.)_**
  [http://www.dcs.gla.ac.uk/Keith/Preface.html](http://www.dcs.gla.ac.uk/Keith/Preface.html)
  A foundational text on the principles of information retrieval systems.

- **Gupta et al. (2014). "Improved pattern learning for bootstrapped entity extraction."**
  Related paper: [Angeli et al. (2014) ACL](https://aclanthology.org/P14-1144)
  Discusses pattern-based bootstrapping approaches to entity extraction.

- **Wang et al. (2019). "GLUE: A Multi-Task Benchmark And Analysis Platform For Natural Language Understanding."**
  [OpenReview (ICLR 2019)](https://openreview.net/forum?id=rJ4km2R5t7) | [GLUE Benchmark](https://gluebenchmark.com)
  Proposes a widely adopted multi-task benchmark for evaluating NLP models.

- **Strubell et al. (2019). "Energy and Policy Considerations for Deep Learning in NLP."**
  [arXiv:1906.02243](https://arxiv.org/abs/1906.02243)
  Investigates the environmental impact of large-scale NLP model training.

- **Dodge et al. (2022). "Software Carbon Intensity (SCI)."**
  [Green Software Foundation](https://greensoftware.foundation/projects/software-carbon-intensity-spec)
  A framework for measuring the carbon intensity of software solutions.

- **Sheng et al. (2019). "The Woman Worked as a Babysitter: On Biases in Language Generation."**
  [arXiv:1904.02264](https://arxiv.org/abs/1904.02264)
  Examines bias in language models via prompts and generated outputs.

- **Dou et al. (2016).**
  [COLING 2016 Paper](https://aclanthology.org/C16-1022)
  Explores multilingual embedding alignment methods to address semantic divergence.

- **Hu et al. (2020). "XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization."**
  [arXiv:2003.11080](https://arxiv.org/abs/2003.11080)
  A benchmark covering a range of cross-lingual transfer tasks.

- **Zinkevich et al. (2022). "Google’s Best Practices for ML Engineering."**
  [Rules of ML – Google Developers](https://developers.google.com/machine-learning/guides/rules-of-ml)
  Offers guidelines on how to design, deploy, and maintain machine learning systems effectively.

These readings offer both historical perspectives and modern insights into how NLP has evolved and why these methods work.

### Practical Components

Finally, you’ll put theory into practice by:

- Building a basic text-processing pipeline (tokenization, cleaning, etc.).
- Implementing your own **TF-IDF** vectorizer or using existing libraries like `scikit-learn`.
- Performing document similarity calculations with sparse vectors.
- Running a simple text classification experiment on a real dataset.
- Comparing **baseline results** with more advanced approaches to see how improvements stack up.
