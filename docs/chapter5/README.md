## 🎓 Course Materials

### 📑 Slides

[Download Session 5 Slides (PDF)](../pdfs/BSE_NLP_Session_5.pdf)

[Download Bonus Slides — Interpreting Embeddings: Cosine vs. L2 (PDF)](../pdfs/BSE_NLP_Session_5_Embeddings.pdf)

### 📓 Notebooks

- [Implementing BERT with Hugging Face Transformers](Session_5_1_BERT_HF_Implementation.ipynb)
- [Visualizing Attention Mechanisms](Session_5_2_Attention_Visualization.ipynb)
- Comparing LSTM vs. BERT vs. TinyBERT vs. ModernBERT _(coming soon)_
- [Interpreting Contextual Embeddings: Anisotropy, Rogue Dimensions & Fixes](Session_5_3_Interpreting_Contextual_Embeddings.ipynb) — [bonus slides](../pdfs/BSE_NLP_Session_5_Embeddings.pdf)

---

## 🚀 Session 5: Attention, Transformers, and BERT

In this fifth session, we move from traditional sequence models to the architecture that revolutionized NLP: the **Transformer**. We analyze how **attention mechanisms** solved the context-length limitations of RNNs, and how **BERT**, built on top of Transformers, became the new backbone of language understanding.

We also explore **fine-tuning** BERT for downstream tasks, and examine several variants (e.g., **SciBERT**, **XLM-T**, **ModernBERT**) tailored for specific domains or efficiency needs.

### 🎯 Learning Objectives

1. Identify the **limitations of RNNs** and understand why attention mechanisms were introduced.
2. Understand the full **Transformer architecture** including self-attention and feed-forward components.
3. Grasp the innovations of **BERT**: bidirectionality, MLM, and NSP.
4. Learn to **fine-tune** BERT for real tasks (NER, classification, QA).
5. Explore extensions and variants like **DistilBERT**, **SciBERT**, **XtremeDistil**, and **ModernBERT**.

---

### 📚 Topics Covered

#### Attention & Transformers

- Limitations of RNNs (sequential processing, long-distance dependencies).
- **Attention Mechanism**: Query-Key-Value, dynamic focus, soft memory.
- **Self-Attention**: Core of the Transformer — all tokens attend to all others.
- **Multi-Head Attention**: Capture different representation subspaces.
- **Transformer Architecture**: Encoder-decoder stack, position encoding, full parallelization.

#### BERT: Bidirectional Encoder Representations from Transformers

- BERT architecture: 12–24 layers, multi-head attention, 110M+ parameters.
- **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**.
- Tokenization strategies (WordPiece, special tokens).
- Fine-tuning BERT for:
  - Classification
  - Token-level tasks (e.g., NER, QA)
- Performance on benchmarks (GLUE, SQuAD).

#### BERT Variants and Extensions

- **SciBERT** for scientific text understanding.
- **EconBERTa** for named entity recognition in economic research.
- **XLM-T** for multilingual social media analysis.
- **XtremeDistilTransformer**: BERT distilled for efficiency.
- **ModernBERT (2024)**: Faster, longer-context, flash attention, rotary embeddings.

---

### 🧠 Key Takeaways

| Architecture     | Sequential? | Long-Context Friendly | Fine-Tunable | Efficient Inference |
|------------------|-------------|------------------------|--------------|----------------------|
| LSTM             | ✅          | ❌                     | ✅           | ⚠️                   |
| Transformer      | ❌          | ✅                     | ✅           | ✅                   |
| BERT             | ❌          | ✅ (but limited tokens) | ✅           | ⚠️                   |
| ModernBERT       | ❌          | ✅ (8k tokens)         | ✅           | ✅                   |

---

### 📖 Bibliography & Recommended Reading

- **Vaswani et al. (2017): Attention Is All You Need** – [Paper](https://arxiv.org/abs/1706.03762)
  The foundation of the Transformer model.

- **Alammar (2018): The Illustrated Transformer** – [Blog Post](https://jalammar.github.io/illustrated-transformer/)
  Highly visual explanation of attention and Transformer layers.

- **Devlin et al. (2019): BERT: Pre-training of Deep Bidirectional Transformers** – [Paper](https://arxiv.org/abs/1810.04805)
  Original BERT paper introducing MLM and NSP.

- **Warner et al. (2024): ModernBERT** – [Paper](https://arxiv.org/pdf/2412.13663)
  A modern rethinking of BERT optimized for efficiency and long-context modeling.

- **Rogers et al. (2020): A Primer in BERTology** – [Paper](https://arxiv.org/abs/2002.12327)
  Analysis and interpretability of BERT’s internal behavior.

- **Bahdanau et al. (2014): Neural Machine Translation by Jointly Learning to Align and Translate** – [Paper](https://arxiv.org/abs/1409.0473)
  The original attention mechanism for sequence-to-sequence models.

- **Liu et al. (2019): RoBERTa: A Robustly Optimized BERT Pretraining Approach** – [Paper](https://arxiv.org/abs/1907.11692)
  Shows BERT was undertrained; longer training and more data improve it.

- **Yang et al. (2019): XLNet: Generalized Autoregressive Pretraining for Language Understanding** – [Paper](https://arxiv.org/abs/1906.08237)
  Permutation language modeling that combines autoregressive and bidirectional context.

- **Lewis et al. (2019): BART: Denoising Sequence-to-Sequence Pre-training** – [Paper](https://arxiv.org/abs/1910.13461)
  Denoising autoencoder pretraining for generation and comprehension.

- **He et al. (2021): DeBERTa: Decoding-enhanced BERT with Disentangled Attention** – [Paper](https://arxiv.org/abs/2006.03654)
  Disentangled content/position attention improving over BERT and RoBERTa.

- **Sanh et al. (2019): DistilBERT, a Distilled Version of BERT** – [Paper](https://arxiv.org/abs/1910.01108)
  A smaller, faster BERT trained by knowledge distillation.

- **Beltagy et al. (2019): SciBERT: A Pretrained Language Model for Scientific Text** – [Paper](https://arxiv.org/abs/1903.10676)
  Domain-adapted BERT for scientific literature.

- **Barbieri et al. (2022): XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis** – [Paper](https://arxiv.org/abs/2104.12250)
  Multilingual social-media transformer for sentiment tasks.

- **Mukherjee et al. (2021): XtremeDistilTransformers: Task Transfer for Task-agnostic Distillation** – [Paper](https://arxiv.org/abs/2106.04563)
  Task-agnostic distillation producing compact multilingual encoders.

- **Lasri et al. (2023): EconBERTa: Towards Robust Extraction of Named Entities in Economics** – [Paper](https://aclanthology.org/2023.findings-emnlp.774/)
  Domain-adapted encoder and the ECON-IE dataset for economics NER.

---

### 💻 Practical Components

- **Hugging Face BERT**: Load, fine-tune, and evaluate BERT on classification or QA tasks.
- **Attention Visualization**: See how attention heads behave using heatmaps and interpret interactions between tokens.
- **Model Benchmarking**: Compare inference time, memory use, and accuracy of LSTM, BERT, TinyBERT, and ModernBERT.
