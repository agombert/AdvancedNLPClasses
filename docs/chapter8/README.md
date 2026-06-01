## 🎓 Course Materials - Practical NLP - 2

## Session 8: Fine-Tuning BERT, Few-Shot Learning with SetFit, and Bias in NLP

In this hands-on session, we explore cutting-edge approaches in **advanced NLP**, including **fine-tuning BERT**, leveraging **few-shot learning with SetFit**, and investigating **biases in NLP models** (like gender biases using WinoGrad schemas).

This notebook is designed as a modular, reusable blueprint for state-of-the-art NLP techniques.

### 📓 Notebooks

* [Session 8: BERT, Few-Shot Learning, and Bias in NLP](Session_8.ipynb)
* [⬇️ Download the student notebook (.ipynb)](Session_8.ipynb)

The student notebook ships with **guided comments only** — fill in the blanks as you go. The corrected version is kept private for the instructor.

---

### 🎯 Learning Objectives

1. **Fine-tune BERT** for text classification with a small dataset.
2. Understand and implement **few-shot learning** using the SetFit framework.
3. **Compare** SetFit with traditional baselines (TF-IDF, fine-tuned BERT) at equal label budget.
4. **Visualize** how SetFit re-shapes the embedding space as labelled data grows from 0 to 64 examples / class.
5. Evaluate models using **standard metrics** (accuracy, precision, recall, F1-score, confusion matrix, ROC/AUC).
6. Analyze and **identify biases** in BERT models using WinoGrad schemas.
7. Discuss **model fairness** and interpretability in modern NLP.

---

### 📚 Topics Covered

#### Fine-Tuning BERT

* DistilBERT with a softmax head on AG News.
* Frozen-encoder vs full fine-tuning comparison.

#### Few-Shot Learning with SetFit

* **Stage 1**: contrastive fine-tuning of a Sentence Transformer on **pairs of labelled
  examples** (positives = same class, negatives = different class).
* **Stage 2**: lightweight **logistic regression** head on top of the fine-tuned embeddings.
* Why **8–64 examples per class** are enough — pair-based blow-up of the training signal.
* Hands-on training of SetFit with 16 examples per class on AG News.
* Apples-to-apples **comparison** with TF-IDF (full data) and DistilBERT (~200 examples).
* **Embedding-space visualization**: PCA(50) → UMAP(2), **fitted once** on the base
  Sentence Transformer, then **transformed** (never refitted) on encoders re-trained
  with 8 / 16 / 32 / 64 examples / class — so movement on the plot is real movement in
  embedding space.
* Optional augmentation of the training set with zero-shot prompt-based labelling.

#### Bias in NLP Models

* WinoGender schemas (Rudinger et al., 2018) to probe gender bias.
* Fill-mask experiments on BERT-large, BERT, and DistilBERT.

---

### 📖 Bibliography & Recommended Reading

* **Hugging Face Transformers Documentation** – [Link](https://huggingface.co/docs/transformers/index)
* **SetFit GitHub repo** – [Link](https://github.com/huggingface/setfit)
* **Tunstall et al. (2022): Efficient Few-Shot Learning Without Prompts** – [Paper](https://arxiv.org/abs/2209.11055)
* **Reimers & Gurevych (2019): Sentence-BERT** – [Paper](https://arxiv.org/abs/1908.10084)
* **Schroff et al. (2015): FaceNet (triplet loss)** – [Paper](https://arxiv.org/abs/1503.03832)
* **WinoGender Schemas** – [Repo](https://github.com/rudinger/winogender-schemas)
* **Rudinger et al. (2018): Gender Bias in Coreference Resolution** – [Paper](https://aclanthology.org/N18-2002/)
* **Fairness in Machine Learning** – [Link](https://fairmlbook.org/)

---

### 💻 Practical Components

* 🏗️ **Fine-tune BERT** on AG News corpus using Hugging Face Transformers.
* 🔄 **Train a SetFit classifier** with as few as **16 examples per class**.
* 📊 **Compare** SetFit vs TF-IDF vs DistilBERT at equal-or-larger label budgets.
* 🌌 **Visualize embedding-space transformations** with PCA + UMAP across multiple SetFit runs.
* 🧪 **Experiment with data augmentation** via prompt-based methods.
* 🕵️‍♂️ **Evaluate model fairness** and **gender bias** in predictions.
* 🎯 **Compare models** using quantitative metrics (ROC/AUC, F1, etc.) and **qualitative outputs** (example-level analysis).
