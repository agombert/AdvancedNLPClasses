## 🎓 Course Materials

### 📑 Slides

[Download Session 10 Slides (PDF)](../pdfs/2025_BSE_NLP_Session_10.pdf)

### 📓 Notebooks

- [LLM with Tools](Session_10_1_LLM_with_Tools.ipynb)
- [LLM as a Judge](Session_10_2_LLM_as_a_Judge.ipynb)
- [ReAct Framework](Session_10_3_ReAct_Framework.ipynb)

---

## 🚀 Session 10: Hallucinations and Agents in Large Language Models

In this session, we explore the challenges of **hallucinations** in LLMs and how to address them. We dive into concepts such as **function calling** and **multi-LLM evaluation** as solutions for reducing errors. Finally, we introduce **agent-based frameworks** (e.g., ReAct) as a next-level strategy for reasoning, planning, and tool use.

We connect theory to practice with real-world examples and Python code for building reliable, dynamic LLM agents.

### 🎯 Learning Objectives

1. Understand the nature of **hallucinations and outdated knowledge** in LLMs.
2. Explore **mitigation strategies**: prompt engineering, retrieval-augmented generation (RAG), function calling, and multi-LLM approaches.
3. Learn how **LLMs as judges** can validate or compare outputs, increasing reliability.
4. Discover how **agents (ReAct framework)** enable iterative planning, reasoning, and tool use.
5. Identify **failure modes** (planning and tool execution errors) in agent-based systems.

---

### 📚 Topics Covered

#### 🌟 Hallucinations and Errors
- **Intrinsic vs. Extrinsic Causes**: From data mismatch to model limitations.
- **Examples**: Factual inaccuracies, outdated knowledge, and misinformation.
- **Consequences**: Misinformation, trust erosion, and practical failures.

#### ⚙️ Mitigation Techniques
- **Prompt Engineering**: Crafting effective prompts to reduce ambiguity.
- **RAG**: Grounding responses with external data.
- **Function Calling**: Using APIs and tools for accurate, real-time answers.
- **Multi-LLM Evaluation**: Using LLMs as judges for quality control.

#### 🤖 Agents and Advanced Use Cases
- **Agent Limitations**: Planning, tool execution, and efficiency challenges.
- **ReAct Framework**: Combining reasoning and acting for iterative solutions.
- **ReAct Steps**: Think → Act → Observe → Repeat.
- **Implementation**: LangChain/LlamaIndex agents in Python for multi-step problem solving.

---

### 🧠 Key Takeaways

| Concept/Technique        | Purpose                           | Benefit                                       |
|--------------------------|-----------------------------------|-----------------------------------------------|
| Hallucination Analysis   | Identify sources of errors       | Better LLM reliability and trustworthiness    |
| Prompt Engineering       | Clear instructions for LLM       | More precise and accurate outputs             |
| RAG & Function Calling   | External knowledge integration   | Reduces hallucinations and outdated answers   |
| LLM-as-a-Judge           | Output validation and ranking    | Automated quality assurance                   |
| Agent Frameworks (ReAct) | Iterative tool-based reasoning   | Handle complex, multi-step tasks effectively  |

---

### 📖 Recommended Reading

- **[Huyen Chip](https://huyenchip.com/2025/01/07/agents.html)** – “Agents”
  A comprehensive guide to agents, including reflection and error correction.

- **[Zheng et al. (2023)](https://arxiv.org/abs/2305.16291)** – “Judging LLM-as-a-Judge”
  Paper exploring how LLMs can reliably score and compare generated outputs.

- **[Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://aclanthology.org/2024.emnlp-main.525.pdf)** – “Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs”
  A paper exploring how to optimize instructions and demonstrations for multi-stage language model programs.

- **LangChain Docs** – [python.langchain.com](https://python.langchain.com)
  Comprehensive toolkit for building LLM agents with tool integration.

- **OpenAI Function Calling Examples** – [openai.com/blog/function-calling](https://openai.com/blog/function-calling)
  How to integrate LLMs with APIs for grounded answers.

- **ReAct Framework** – [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629)
  Foundation paper introducing the ReAct agent approach.

---

### 💻 Practical Components

- **LLM Tools Integration**: Function calling examples for real-time data.
- **LLM-as-a-Judge**: Code snippets for output ranking and quality control.
- **ReAct Agent Implementation**: LangChain-based examples with external tool usage.
