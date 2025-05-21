## 🎓 Course Materials

### 📑 Slides

[Download Session 7 Slides (PDF)](../pdfs/2025_BSE_NLP_Session_7.pdf)

### 📓 Notebooks

* [Detecting Gender Bias in LLMs with Prompting](Session_7_1_Gender_Bias_Detection.ipynb)
* [Evaluating Cross-Linguistic Fairness in Classification](Session_7_2_Cross_Language_Evaluation.ipynb)
* [Reduce the size of a BERT model](Session_7_3_reduce_BERT_model.ipynb)

---

## ⚖️ Session 7: Injustice and Biases in NLP

In this session, we investigate one of the most pressing ethical issues in NLP: **biases in language models** and the broader implications of deploying LLMs in socially sensitive contexts.

We study where these biases come from, how they manifest, and what we can do to **detect, mitigate, and monitor** them — with a particular focus on **Large Language Models** like BERT and GPT.

We also explore the **environmental costs** of modern NLP, promoting not just fairness in output, but fairness in who pays the cost of progress.

---

### 🎯 Learning Objectives

1. Understand the different types of biases present in NLP systems.
2. Analyze real-world harms caused by bias in language technologies.
3. Explore how biases arise during training and deployment of LLMs.
4. Learn how to detect bias using statistical, adversarial, and prompt-based techniques.
5. Implement practical mitigation strategies: pre-, mid-, and post-training.
6. Understand the **ecological footprint** of LLMs and low-resource alternatives.

---

### 📚 Topics Covered

#### 🧠 Foundations of Bias in NLP

* Historical and societal roots of bias in AI.
* Linguistic and cultural overrepresentation.
* Gender, racial, and socioeconomic stereotyping in LLMs.
* The "Stochastic Parrot" critique (Bender et al., 2021).

#### 🔍 Detection Strategies

* **Statistical Fairness Criteria**: Independence and separation metrics.
* **Prompt-based Bias Testing**: e.g., Sheng et al. (2019) templates.
* **Sentiment Disparities**: Analyzing polarity across demographic descriptors.
* **Occupation Prediction Bias**: Kirk et al. (2021) methodology.

#### 🛠️ Mitigation Approaches

* **Pre-training**: Balanced datasets, multilingual corpora (e.g., BLOOM).
* **During Training**: Fairness-aware loss functions (Chuang et al., 2021).
* **Post-training**:

  * Self-debiasing (Schick et al., 2021).
  * Neural editing (Suau et al., 2022).

#### 🌍 Environmental Impacts

* Carbon footprint of LLMs (Strubell et al., Luccioni et al.)
* Model compression techniques:

  * Distillation (Hinton et al., 2015)
  * Quantization
  * Pruning

---

### 🧠 Key Takeaways

| Topic                       | Risk/Concern                        | Mitigation Strategy                         |
| --------------------------- | ----------------------------------- | ------------------------------------------- |
| Gender/Racial Bias          | Reinforces stereotypes              | Prompt analysis, fairness-aware training    |
| Linguistic Inequality       | Language exclusion                  | Multilingual training, inclusive benchmarks |
| Coherence vs. Understanding | Fluent but biased/misleading output | Self-diagnosis and auditing tools           |
| Ecological Impact           | High energy & emissions             | Distillation, quantization, pruning         |

---

### 📖 Bibliography & Recommended Reading

* **The Social Dilemma** – [Documentary](https://www.imdb.com/fr/title/tt11464826/)
* **Bender et al. (2021): On the Dangers of Stochastic Parrots** – [Paper](https://dl.acm.org/doi/10.1145/3442188.3445922)
* **Blodgett et al. (2020): Language (Technology) is Power** – [Paper](https://arxiv.org/abs/2005.14050)
* **Sheng et al. (2019): The Woman Worked as a Babysitter** – [Paper](https://arxiv.org/abs/1903.03862)
* **Kirk et al. (2021): Bias in GPT Occupational Predictions** – [Paper](https://arxiv.org/abs/2105.05596)
* **Chuang et al. (2021): Fairness Constraints in Loss** – [Paper](https://arxiv.org/abs/2109.05211)
* **Schick et al. (2021): Self-Diagnosis and Debiasing** – [Paper](https://arxiv.org/abs/2103.00453)
* **Suau et al. (2022): Neuron-Level Bias Mitigation** – [Paper](https://arxiv.org/abs/2201.11714)
* **Strubell et al. (2019): Energy and Policy Considerations for Deep NLP** – [Paper](https://arxiv.org/abs/1906.02243)
* **Luccioni et al. (2023): Carbon Footprint of BLOOM** – [Paper](https://arxiv.org/abs/2211.02001)

---

### 💻 Practical Components

* **Prompt-Based Bias Detection**: Use controlled sentence templates to assess gender and racial stereotypes in text generation.
* **Cross-Language Model Evaluation**: Compare model predictions across languages to quantify linguistic fairness.
* **Reduce the size of a BERT model**: Use distillation, quantization, and pruning to reduce the size of a BERT model.
