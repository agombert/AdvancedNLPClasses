## 🎓 Course Materials

### 📑 Slides

[Download Session 9 Slides (PDF)](../pdfs/BSE_NLP_Session_9.pdf)

### 📓 Notebooks

- [Prompt Engineering](Session_9_1_Prompt_Engineering.ipynb)
- [RAG](Session_9_2_RAG.ipynb)

---

## 🚀 Session 9: Large Language Models (LLMs) Basics

In this ninth session, we dive into the world of **Large Language Models (LLMs)** — from the foundations of ChatGPT to advanced prompt engineering and innovative training techniques like LoRA and QLoRA.

We explore the **training processes** (SFT, RM, PPO), how models like **GitHub Copilot and Cursor** boost productivity, and advanced topics like **Retrieval-Augmented Generation (RAG)**.

### 🎯 Learning Objectives

1. Understand the key components of **LLM training**: Supervised Fine-Tuning (SFT), Reward Modeling (RM), Proximal Policy Optimization (PPO), and **Direct Preference Optimization (DPO)**.
2. Understand **reasoning models** (o1, DeepSeek-R1) and how they are trained with **RL on verifiable rewards (RLVR / GRPO)**.
3. Learn about **prompt engineering, few-shot learning**, structured outputs, and the **temperature vs. top-p** trade-off.
4. Build a complete **RAG pipeline** (chunk → embed → retrieve → rerank → generate) and learn how to **evaluate** it.
5. Distinguish **continued pre-training** (inject knowledge) from **SFT** (shape behavior), and grasp how **LoRA** enables fine-tuning large models on a single GPU.
6. Connect LLM-powered tools like **Cursor, LlamaIndex**, and **Vera** to real-world applications.

---

### 📚 Topics Covered

#### 🔧 Training & Aligning LLMs
- **Supervised Fine-Tuning (SFT)**: Aligning models to conversational data.
- **Reward Modeling (RM)** & **PPO**: Learning from and optimizing for human feedback.
- **Direct Preference Optimization (DPO)**: Preference as a log-ratio comparison → binary classification, no reward model.
- **The preference-optimization zoo**: IPO, ORPO, SimPO, KTO, RLAIF, GRPO.
- **Limitations without RLHF**: Overfitting and bias risks.

#### 🧠 Reasoning Models
- **o1 / DeepSeek-R1**: Learned chain of thought and test-time compute.
- **RLVR**: RL on verifiable rewards (math = exact match, code = unit tests).
- **The R1 pipeline**: cold start → reasoning RL → data factory (~800k) → final RL; distillation to small models.

#### 💡 Applications of LLMs
- **Cursor**: AI-powered code editor for efficient development.
- **LlamaIndex**: RAG for internal data search.
- **Vera**: Fact-checking for public trust.

#### ⚙️ Prompt Engineering
- **Techniques**: Few-shot, chain of thoughts, schema-constrained / structured outputs (`response_format` + Pydantic).
- **Sampling Parameters**: Temperature vs. top-p, which one to tune and why.

#### 🔎 Retrieval-Augmented Generation (RAG)
- **Naive RAG**: Retrieve → concatenate → generate.
- **The full pipeline**: chunk → embed → vector store → retrieve → rerank → generate.
- **Reranking**: Cheap recall (bi-encoder) then precise ranking (cross-encoder).
- **Evaluation**: Retrieval metrics (Recall@k, Precision@k, MRR, nDCG) and generation metrics (faithfulness, answer relevance) via RAGAS / LLM-as-a-judge.
- **Advanced retrieval**: HyDE & query transformation.

#### 🎯 Fine-Tuning & Model Adaptation
- **Continued Pre-Training (CPT)**: Inject knowledge with next-token prediction; protocols (LR re-warm/re-decay, ~5% replay) and token budgets by model size.
- **SFT**: Shape behavior with prompt → response pairs.
- **Training Complexity**: Challenges of training a 7B parameter model.
- **LoRA**: Low-Rank Adaptation for single-GPU fine-tuning.

#### 🌟 UX & Human Feedback
- **Copilot Example**: Integrating UX for continuous learning.
- **User feedback**: Passive vs. active feedback collection.

---

### 🧠 Key Takeaways

| Component/Technique | Purpose                            | Benefit                                    |
|---------------------|------------------------------------|--------------------------------------------|
| SFT                 | Initial conversational fine-tuning | Groundwork for aligned responses           |
| RM + PPO            | Score and optimize via humans      | Aligns LLM outputs with human preferences  |
| DPO                 | Preference as a binary classifier  | No reward model, more stable than PPO      |
| RLVR / GRPO         | RL on verifiable rewards           | Trains reasoning models (math, code)       |
| CPT                 | Continued pre-training on raw text | Injects domain knowledge before SFT        |
| LoRA                | Efficient fine-tuning              | Train large models on a single GPU         |
| Prompt Engineering  | Guide LLM outputs                  | More accurate, context-aware interactions  |
| RAG + Reranking     | External knowledge retrieval       | Boosts factuality, fewer hallucinations    |

---

### 📖 Recommended Reading

- **[Rafailov et al. (2023)](https://arxiv.org/abs/2305.18290)**: “Direct Preference Optimization: Your Language Model is Secretly a Reward Model”
  Paper introducing DPO; reframes preference optimization as a simple classification loss.
- **Mantis NLP / Argilla Blog Posts**: [argilla.io/blog](https://argilla.io/blog)
  Hands-on articles on DPO, ORPO, SimPO, KTO, and other preference-optimization methods.
- **[DeepSeek-AI (2025)](https://arxiv.org/abs/2501.12948)**: “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning”
  Paper behind R1 / R1-Zero: pure RL, verifiable rewards, the cold-start + data-factory pipeline, and distillation.
- **[Jay Alammar – The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)**
  Visual walkthrough of R1, including the rule-based verification of generated code.
- **[Ibrahim & Gupta et al. (2024)](https://arxiv.org/abs/2403.08763)**: “Simple and Scalable Strategies to Continually Pre-train Large Language Models”
  LR re-warming/re-decaying and ~5% replay to avoid catastrophic forgetting.
- **[Chen et al. (2025)](https://aclanthology.org/2025.acl-long.289.pdf)**: “Towards Effective and Efficient Continual Pre-training of Large Language Models” (Llama-3-SynE)
  Data-mixture (1:7:2), perplexity-based curriculum, and token budgets for domain CPT.
- **[Rozière et al. (2023)](https://arxiv.org/abs/2308.12950)**: “Code Llama: Open Foundation Models for Code”
  CPT example: Llama 2 continued on 500B tokens of code.
- **[Azerbayev et al. (2023)](https://arxiv.org/abs/2310.10631)**: “Llemma: An Open Language Model for Mathematics”
  CPT example: Code Llama continued on 200B math tokens (Proof-Pile-2).
- **[Es et al. (2023)](https://arxiv.org/abs/2309.15217)**: “RAGAS: Automated Evaluation of Retrieval Augmented Generation”
  Faithfulness / answer-relevance / context metrics, often scored with an LLM-as-a-judge.
- **Gao et al. (2022)**: “Precise Zero-Shot Dense Retrieval without Relevance Labels”
  Paper introducing Hypothetical Document Embeddings (HyDE).
- **Hu et al. (2021)**: "LoRA: Low-Rank Adaptation of Large Language Models"
  Paper introducing LoRA.
- **[Wei et al. (2022)](https://arxiv.org/abs/2201.11903)**: “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models”
  Shows that intermediate reasoning steps unlock multi-step problem solving.
- **[Shin et al. (2021)](https://aclanthology.org/2021.emnlp-main.608/)**: “Constrained Language Models Yield Few-Shot Semantic Parsers”
  Uses constrained decoding to turn LLMs into few-shot semantic parsers.
- **[Tie et al. (2025)](https://arxiv.org/abs/2503.06072)**: “A Survey on Post-training of Large Language Models”
  Survey charting the evolution of alignment, fine-tuning, and reasoning methods.
- **[Liu et al. (2023)](https://arxiv.org/abs/2310.02170)**: “A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration (DyLAN)”
  Multi-agent collaboration framework with automatic team optimization.
- **Blog post** - [blog.eleuther.ai/transformer-math/](https://blog.eleuther.ai/transformer-math/)
  Blog post explaining the math behind LLM computation.
- **Blog post** - [kipp.ly/transformer-inference-arithmetic/](https://kipp.ly/transformer-inference-arithmetic/)
  Blog post explaining the arithmetic behind LLM inference.

---

### 💻 Practical Components

- **Code Examples**: Fine-tuning with LoRA on single GPUs.
- **Prompt Engineering Snippets**: Showcasing few-shot learning and chain-of-thought examples.
- **OpenAI API Usage**: Python code for prompt optimization and few-shot prompting.
