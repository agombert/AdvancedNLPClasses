## 🎓 Course Materials

### 📑 Slides

[Download Session 6 Slides (PDF)](../pdfs/2025_BSE_NLP_Session_6.pdf)

### 📓 Notebooks

- [Topic Modeling with BERTopic](Session_6_1_BERTopic_Topic_Modeling.ipynb)
- [Zero-Shot Classification with BERT/RoBERTa/DistilBERT](Session_6_2_Zero_Shot_Classification.ipynb)
- [Generating Movie Reviews with GPT Prompting](Session_6_3_Generation_with_GPT.ipynb)

---

## 🚀 Session 6: Few-Shot Learning with BERT Models

In this sixth session, we dive into **Few-Shot** and **Zero-Shot Learning** in NLP. These techniques are designed to work in **data-scarce environments**, mimicking how humans can generalize from just a few examples.

We explore the remarkable generalization abilities of **BERT-like models**, learn how to apply **zero-shot classification** using simple prompting techniques, and discover how to **generate synthetic data** with generative models like **GPT-2**. We also investigate state-of-the-art techniques like **SetFit** that combine contrastive learning and fine-tuning — achieving strong results with minimal data.

### 🎯 Learning Objectives

1. Understand the motivations for Zero-Shot and Few-Shot Learning in NLP.
2. Explore how BERT and Transformer-based models naturally support these paradigms.
3. Apply different approaches to zero/few-shot classification, including **NLI**, **cloze prompting**, and **embedding similarity**.
4. Learn to generate task-specific labeled data using **GPT prompting**.
5. Fine-tune Sentence Transformers with contrastive learning using **SetFit**.

---

### 📚 Topics Covered

#### Few-Shot Learning Foundations

- Data scarcity challenges in real-world NLP.
- Human-like generalization: learning from just a few examples.
- Why BERT-like models are ideal for few-shot learning.

#### Zero-Shot Classification Techniques

- **Latent Embedding Matching**: Use similarity between sentence and class embeddings.
- **Natural Language Inference (NLI)**: Frame classification as premise-hypothesis inference.
- **Cloze Task with BERT**: Convert classification to fill-in-the-blank prediction.
- **Weak Supervision with Snorkel**: Labeling via noisy heuristics.

#### Prompt Engineering and Text Generation

- Use **GPT-2 or GPT-3** to generate balanced synthetic datasets.
- Prompting as a tool for classification, style transfer, and data augmentation.

#### Advanced Few-Shot Learning

- **iPET** and **Pattern-Exploit Training** (Schick & Schütze, 2020).
- **SetFit (Tunstall et al., 2022)**:
  - Few-shot learning via contrastive training of Sentence Transformers.
  - No need for full finetuning or large hardware.
  - Very fast and cost-efficient.

---

### 🧠 Key Takeaways

| Approach               | Data Required | Training Time | Interpretability |
|------------------------|---------------|---------------|------------------|
| Traditional Supervised | High          | Long          | ✅                |
| Zero-Shot (NLI/Embeds) | None          | None          | ✅                |
| Cloze Prompting        | None          | None          | ⚠️                |
| GPT-based Generation   | None          | Medium        | ❌                |
| SetFit (Contrastive)   | Very Low (8–16 examples) | Very Fast | ✅         |

---

### 📖 Bibliography & Recommended Reading

- **Brown et al. (2020): Language Models are Few-Shot Learners** – [Paper](https://arxiv.org/abs/2005.14165)
  The GPT-3 paper showing incredible few-shot generalization.

- **SetFit: Efficient Few-Shot Learning Without Prompts** - [Blog](https://huggingface.co/blog/setfit)
  A simple framework for contrastive learning of visual representations.

- **Zero-Shot Text Classification** - [Blog](https://joeddav.github.io/blog/2020/05/29/ZSL.html)
  A comprehensive guide to zero-shot classification.

- **DINO:Using Big Language Models To Generate Entire Datasets From Scratch** - [Blog](http://timoschick.com/research/2021/05/19/dino.html)
  Using a LLM to generate a dataset from scratch.

- **Schick & Schütze (2020): Exploiting Cloze Questions for Few-Shot Text Classification** – [Paper](https://arxiv.org/abs/2001.07676)
  Introduction to iPET and pattern-based classification using BERT.

- **Tunstall et al. (2022): Efficient Few-Shot Learning with Sentence Transformers** – [Paper](https://arxiv.org/abs/2209.11055)
  SetFit, a scalable, contrastive learning approach to few-shot classification.

- **Yin et al. (2019): Benchmarking Zero-shot Text Classification** – [Paper](https://arxiv.org/abs/1909.00161)
  Comparative analysis of zero-shot approaches including NLI and embeddings.

- **Snorkel: Weak Supervision for Training Data** – [Website](https://www.snorkel.org)
  Framework to label data using programmatic rules and heuristics.

---

### 💻 Practical Components

- **Topic Modeling with BERTopic**: Use embeddings and clustering to explore topic structures in reviews.
- **Zero-Shot Classification**: Leverage Hugging Face pipelines with BERT/RoBERTa/DistilBERT for inference-only classification.
- **Prompting with GPT-2**: Learn to generate realistic and diverse movie reviews using carefully crafted prompts.
