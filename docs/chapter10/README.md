## 🎓 Course Materials

### 📑 Slides

[Download Session 10 Slides (PDF)](../pdfs/BSE_NLP_Session_10.pdf)

### 📓 Notebooks

- [LLM with Tools](Session_10_1_LLM_with_Tools.ipynb)
- [LLM as a Judge](Session_10_2_LLM_as_a_Judge.ipynb)
- [ReAct Framework](Session_10_3_ReAct_Framework.ipynb)

---

## 🚀 Session 10: Tools, Agents & the Limits of LLMs

In this session we move from a frozen, one-shot LLM to a system that can **act on its own**. We give the model **tools** (function / tool calling), use LLMs as a **feedback loop** to improve themselves (automated prompt engineering and LLM-as-a-judge), standardize tool access with **MCP**, and chain reasoning and acting into **agents** (ReAct and the modern agent toolbox). We then spend a substantial part of the session on the **limits**: hallucinations and why they happen, shaky reasoning, bias, compounding errors, and the human cost of delegating, each grounded in recent research.

### 🎯 Learning Objectives

1. Explain **tool / function calling** and implement it with the **modern OpenAI Responses API**.
2. Use LLMs as a **feedback loop**: **Automated Prompt Engineering** (APE, OPRO) and **LLM-as-a-judge** to improve prompts and pipelines.
3. Recognize and **debias** an LLM judge (position, verbosity, self-preference, leniency biases).
4. Understand **MCP** (Model Context Protocol) as the standard plug that turns N×M tool integrations into N+M.
5. Build **agents** with **ReAct** and compose modern patterns (reflection, planning, multi-agent), with concrete examples (Claude Code, a deep-research agent).
6. Explain **why agents compound errors** over long horizons, and **why LLMs hallucinate** from the training objective and decoding.
7. Critically assess LLM **reasoning, bias, and societal impact** through recent peer-reviewed studies.

---

### 📚 Topics Covered

#### 🛠️ Tools: Giving LLMs the Ability to Act
- **Why tools**: the LLM stops being the source of truth and becomes the orchestrator that fetches it.
- **The tool-calling loop**: declare → decide → execute (your code) → respond, grounded.
- **Modern API**: the **Responses API** (`client.responses.create`, flat tool schema, `function_call_output` + `call_id`), the current default over Chat Completions.

#### 🔁 LLMs as a Feedback Loop
- **Automated Prompt Engineering (APE)**: prompts as a search problem, propose → execute → score → resample (up to +8% GSM8K). OPRO (a (prompt, score) trajectory) and DSPy as alternatives.
- **LLM-as-a-Judge**: pointwise vs. pairwise vs. reference-guided scoring as the `evaluate()` signal in the loop; prefer closed comparisons over 1-5 scores.
- **Judge-built datasets**: pairwise judgments also generate high-quality preference data (UltraFeedback).
- **Judge biases**: position, verbosity, self-preference, sycophancy, and concrete fixes (order-swap, length control, cross-model judge, rubric).
- **Self-improving pipeline**: optimizer + target + judge LLMs interacting as tools, with a human-checked held-out set as ground truth.

#### 🔌 MCP: a Standard Plug for Tools
- **The N×M problem**: bespoke connectors for every (model, tool) pair do not scale.
- **Model Context Protocol**: one open standard (JSON-RPC), write an integration once, every client uses it (N+M).
- **Three primitives**: tools (side effects), resources (read-only data), prompts (templates).
- **Ecosystem**: 500+ servers (GitHub, Postgres, Slack, Figma, ...), adopted by Anthropic/OpenAI/Google, stewarded by the Linux Foundation.

#### 🤖 Agents
- **From a single tool call to an agent**: plan, act, observe, adapt, stop, with no human in the inner loop.
- **ReAct**: interleaving reasoning and acting (Think → Act → Observe → Repeat).
- **The modern agent toolbox**: reflection/Reflexion, planning (ReWOO), CodeAct, multi-agent (supervisor-worker), evaluator-optimizer.
- **Examples**: Claude Code (gather → act → verify in the terminal) and a deep-research agent (plan → search → reflect → synthesize).
- **Implementation**: modern LangGraph `create_react_agent`.
- **Compounding errors**: per-step accuracy multiplies (0.85^10 ≈ 20%), success collapses on long-horizon tasks.

#### ⚠️ The Limits of LLMs
- **Hallucinations**: faithfulness vs. factuality errors, with up-to-date examples.
- **Why they happen**: the likelihood objective rewards plausibility not truth, no "I don't know" gradient, decoding randomness, data gaps.
- **Reasoning under pressure**: insensitivity to meaning, brittle analogies, weak rigorous proofs.
- **Bias**: implicit bias surviving explicit fairness tests, and cultural bias baked in by data and tokenization.
- **Human cost**: productivity gains vs. losses by expertise, delegation eroding skill, honesty, and institutions.
- **Security**: jailbreaks vs. prompt injection (direct and indirect), with real attacks (EchoLeak zero-click exfiltration; the "Comment & Control" injection hijacking coding agents in GitHub Actions to leak secrets). Untrusted content the agent reads is a hostile attack surface.

---

### 📖 Recommended Reading

- **[Zhou et al. (2022)](https://arxiv.org/abs/2211.01910)**: "Large Language Models Are Human-Level Prompt Engineers"
  Introduces APE: automatic instruction generation and selection as black-box optimization.
- **[Yang et al. (2023)](https://arxiv.org/abs/2309.03409)**: "Large Language Models as Optimizers"
  OPRO: the LLM refines prompts from a trajectory of past prompts and scores (+8% GSM8K, +50% BBH).
- **[Zheng et al. (2023)](https://arxiv.org/abs/2306.05685)**: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
  LLM judges align with humans over 80% of the time, and documents position, verbosity, and self-preference biases.
- **[Cui et al. (2023)](https://arxiv.org/abs/2310.01377)**: "UltraFeedback: Boosting Language Models with High-quality Feedback"
  Uses GPT-4 judgments to build ~340k preference pairs, an LLM-as-a-judge turned into a dataset generator.
- **[OpenAI Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses)**: migration guide
  Current default API for tool calling and agentic loops, flat tool schema and `call_id`-linked outputs.
- **[ReAct: Yao et al. (2022)](https://arxiv.org/pdf/2210.03629)**: "ReAct: Synergizing Reasoning and Acting in Language Models"
  Foundation paper for the reason-and-act agent loop.
- **[LangGraph Docs](https://langchain-ai.github.io/langgraph/)**: building agents
  Current path for ReAct agents (`create_react_agent`), replacing the deprecated `initialize_agent`.
- **[Model Context Protocol](https://modelcontextprotocol.io/)**: the open standard for tool access
  JSON-RPC protocol exposing tools, resources, and prompts, the "USB-C for AI" that turns N×M into N+M.
- **[Anthropic (2024)](https://www.anthropic.com/news/model-context-protocol)**: "Introducing the Model Context Protocol"
  Announcement and rationale for MCP, the standard agents use to reach external systems.
- **[Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)**: the gather → act → verify loop
  How Claude Code structures the agent loop, its core tools, and MCP integration.
- **[Zhu et al. (2025)](https://arxiv.org/abs/2509.25370)**: "Where LLM Agents Fail and How They Can Learn From Failures"
  Error propagation and compounding failures in long-horizon agentic tasks.
- **[Petrov et al. (2025)](https://arxiv.org/abs/2503.21934)**: "Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad"
  Best model ~25%, all others below 5% on rigorous proofs, despite strong answer-only scores.
- **[Lewis & Mitchell (2025)](https://arxiv.org/abs/2411.14215)**: "Evaluating the Robustness of Analogical Reasoning in Large Language Models" (TMLR)
  Analogy performance collapses under novel variants (e.g. fictional alphabets) while humans stay robust.
- **[Scientific Reports (2024)](https://www.nature.com/articles/s41598-024-79531-8)**: "Testing AI on language comprehension tasks reveals insensitivity to underlying meaning"
  LLMs perform at chance on meaning-probing tasks and waver under minor rephrasings (26,680 datapoints).
- **[Bai et al. (2025)](https://www.pnas.org/doi/10.1073/pnas.2416228122)**: "Explicitly unbiased large language models still form biased associations" (PNAS)
  8 aligned models show implicit stereotypes and biased decisions despite passing explicit bias tests.
- **[Naous & Xu (2025)](https://aclanthology.org/2025.naacl-long.326/)**: "On the Origin of Cultural Biases in Language Models" (NAACL)
  CAMeL-2 (58,086 entities) traces a Western-culture default to pre-training data and tokenization.
- **[METR (2025)](https://arxiv.org/abs/2507.09089)**: "Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity"
  Experienced devs were 19% slower with AI while believing they were 20% faster.
- **[Cui et al. (2026)](https://pubsonline.informs.org/doi/10.1287/mnsc.2025.00535)**: "The Effects of Generative AI on High-Skilled Work" (Management Science)
  Three RCTs, 4,867 developers, +26% tasks completed, with the largest gains for less experienced developers.
- **[Köbis et al. (2025)](https://www.nature.com/articles/s41586-025-09505-x)**: "Delegation to artificial intelligence can increase dishonest behaviour" (Nature)
  Delegating reporting to AI collapses honesty from 95% to 12-16%, machines comply with unethical instructions more than humans.
- **[Hartzog & Silbey (2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5870623)**: "How AI Destroys Institutions" (UC Law Journal, forthcoming)
  Legal/societal argument that AI affordances erode expertise, short-circuit deliberation, and isolate people.
- **[EchoLeak, CVE-2025-32711](https://arxiv.org/abs/2509.10540)**: "EchoLeak: The First Real-World Zero-Click Prompt Injection Exploit in a Production LLM System"
  Indirect prompt injection in M365 Copilot: a crafted email exfiltrates data via an auto-fetched image, no user click.
- **[Guan et al. (2026), "Comment & Control"](https://www.securityweek.com/claude-code-gemini-cli-github-copilot-agents-vulnerable-to-prompt-injection-via-comments/)**: prompt injection hijacking coding agents via GitHub comments
  One pattern broke Claude Code, Gemini CLI, and GitHub Copilot in GitHub Actions: a PR title/comment injects the auto-reviewer, which leaks `ANTHROPIC_API_KEY` / `GITHUB_TOKEN` back through a PR comment or the Actions log.
- **[Huyen Chip (2025)](https://huyenchip.com/2025/01/07/agents.html)**: "Agents"
  Practical guide to agents, reflection, and error correction.

---

### 💻 Practical Components

- **Tool Calling**: Responses API examples for grounded, real-time answers.
- **Automated Prompt Engineering**: an OPRO-style optimization loop driven by an LLM judge.
- **LLM-as-a-Judge**: debiased pairwise scoring (order-swap) as the evaluation signal.
- **MCP**: connecting an agent to external tools through Model Context Protocol servers.
- **ReAct Agent Implementation**: LangGraph-based agent with external tools.
