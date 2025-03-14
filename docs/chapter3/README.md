# Word Embeddings

## Session 3: Word Representation in Vector Space

This session explores the concept of word embeddings, which revolutionized NLP by representing words as dense vectors in a continuous vector space, capturing semantic relationships between words.

### Learning Objectives

- Understand the limitations of traditional one-hot encoding for word representation
- Learn about distributional semantics and the theoretical foundations of word embeddings
- Explore different word embedding techniques and their properties
- Gain practical experience working with pre-trained and custom word embeddings
- Understand the evolution from static to contextual word embeddings

### Topics Covered

#### Static Word Embeddings

- Word2Vec (Skip-gram and CBOW architectures)
- GloVe (Global Vectors for Word Representation)
- FastText and subword information
- Evaluation methods for word embeddings
- Handling out-of-vocabulary words
- Applications of static word embeddings

#### Contextual Embeddings

- Limitations of static word embeddings
- Introduction to contextual word representations
- ELMo (Embeddings from Language Models)
- Early BERT embeddings
- Comparing static vs. contextual embeddings
- Transfer learning with pre-trained embeddings

### Recommended Reading

- Mikolov et al. (2013) "Efficient Estimation of Word Representations in Vector Space"
- Pennington et al. (2014) "GloVe: Global Vectors for Word Representation"
- Bojanowski et al. (2016) "Enriching Word Vectors with Subword Information"
- Peters et al. (2018) "Deep contextualized word representations"
- Alamar, Jay (2018) "The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)"

### Practical Components

- Training custom Word2Vec embeddings on domain-specific corpora
- Using pre-trained embeddings from libraries like Gensim
- Visualizing word embeddings using dimensionality reduction techniques
- Solving word analogy tasks with vector arithmetic
- Implementing simple NLP tasks using both static and contextual embeddings
