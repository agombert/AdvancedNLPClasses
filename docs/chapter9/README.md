## 🎓 Course Materials

### 📑 Slides

[Download Session 9 Slides (PDF)](../pdfs/2025_BSE_NLP_Session_9.pdf)

### 📓 Notebooks

- [Prompt Engineering](Session_9_1_Prompt_Engineering.ipynb)
- [RAG](Session_9_2_RAG.ipynb)
- [LoRA](Session_9_3_LoRA.ipynb)

---

## 🚀 Session 9: Large Language Models (LLMs) Basics

In this ninth session, we dive into the world of **Large Language Models (LLMs)** — from the foundations of ChatGPT to advanced prompt engineering and innovative training techniques like LoRA and QLoRA.

We explore the **training processes** (SFT, RM, PPO), how models like **GitHub Copilot and Cursor** boost productivity, and advanced topics like **Retrieval-Augmented Generation (RAG)**.

### 🎯 Learning Objectives

1. Understand the key components of **LLM training**: Supervised Fine-Tuning (SFT), Reward Modeling (RM), and Proximal Policy Optimization (PPO).
2. Learn about **prompt engineering, few-shot learning**, and structured output generation.
3. Analyze the **advantages and challenges** of RAG and why hypothetical document embeddings (HyDE) can overcome limitations.
4. Grasp how **fine-tuning (LoRA, QLoRA)** enables adapting large models on a single GPU.
5. Connect LLM-powered tools like **Cursor, LlamaIndex**, and **Vera** to real-world applications.

---

### 📚 Topics Covered

#### 🔧 Training LLMs
- **Supervised Fine-Tuning (SFT)**: Aligning models to conversational data.
- **Reward Modeling (RM)**: Learning from human feedback.
- **Proximal Policy Optimization (PPO)**: Aligning outputs with preferences.
- **Direct Preference Optimization (DPO), SimPO**: Evolving preference-optimization methods.
- **Limitations without RLHF**: Overfitting and bias risks.

#### 💡 Applications of LLMs
- **Cursor**: AI-powered code editor for efficient development.
- **LlamaIndex**: RAG for internal data search.
- **Vera**: Fact-checking for public trust.

#### ⚙️ Maximizing LLM Potential
- **Prompt Engineering**: Techniques (few-shot, chain of thoughts, schema-constrained outputs).
- **Sampling Parameters**: Temperature, top-p sampling trade-offs.
- **Retrieval-Augmented Generation (RAG)**: Naive vs. advanced retrieval strategies.
- **Hypothetical Document Embeddings (HyDE)**: Solving RAG’s limitations.

#### 🎯 Model Adaptation on Limited Hardware
- **Training Complexity**: Challenges of training a 7B parameter model.
- **LoRA**: Low-Rank Adaptation for single-GPU fine-tuning.
- **QLoRA**: Quantized LoRA for reduced memory footprint.

#### 🌟 UX & Human Feedback
- **Copilot Example**: Integrating UX for continuous learning.
- **User feedback**: Passive vs. active feedback collection.

---

### 🧠 Key Takeaways

| Component/Technique | Purpose                           | Benefit                                    |
|---------------------|-----------------------------------|--------------------------------------------|
| SFT                 | Initial conversational fine-tuning | Groundwork for aligned responses           |
| RM                  | Score LLM responses via humans    | Learn to rank and prefer high-quality text |
| PPO/DPO/SimPO       | Optimize with preferences         | Aligns LLM outputs with human needs        |
| LoRA                | Efficient fine-tuning             | Train large models on a single GPU         |
| Prompt Engineering  | Guide LLM outputs                 | More accurate, context-aware interactions  |
| RAG/HyDE            | External knowledge retrieval      | Boosts factuality and relevance            |

---

### 📖 Recommended Reading

- **Argilla Blog Posts** – [argilla.io/blog](https://argilla.io/blog)
  Insightful articles on DPO, ORPO, SimPO, and other advanced LLM training methods.
- **Gao et al. (2022)** – “Precise Zero-Shot Dense Retrieval without Relevance Labels”
  Paper introducing Hypothetical Document Embeddings (HyDE).
- **Hu et al. (2021)** - "LoRA: Low-Rank Adaptation of Large Language Models"
  Paper introducing LoRA.
- **[Wei et al. (2022)](https://arxiv.org/abs/2201.11903)** – “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models”
  Shows that intermediate reasoning steps unlock multi-step problem solving.
- **[Shin et al. (2021)](https://aclanthology.org/2021.emnlp-main.608/)** – “Constrained Language Models Yield Few-Shot Semantic Parsers”
  Uses constrained decoding to turn LLMs into few-shot semantic parsers.
- **[Tie et al. (2025)](https://arxiv.org/abs/2503.06072)** – “A Survey on Post-training of Large Language Models”
  Survey charting the evolution of alignment, fine-tuning, and reasoning methods.
- **[Liu et al. (2023)](https://arxiv.org/abs/2310.02170)** – “A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration (DyLAN)”
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
