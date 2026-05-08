## Recommended Materials


### A few newsletters/blogs on NLP and AI you can subscribe to

- [Sebastian Ruder: NLP News](https://newsletter.ruder.io/): Deep dives and curated highlights on the latest NLP research, models, and trends.
- [Melanie Mitchell: AI: A Guide for Thinking Humans](https://aiguide.substack.com/): Critical, accessible essays on AI progress, reasoning, and the limits of current models.
- [Jay Alammar: Language Models & Co.](https://newsletter.languagemodels.co/): Visual, intuitive explanations of LLMs, transformers, and applied NLP concepts.
- [Sebastian Raschka: Ahead of AI](https://magazine.sebastianraschka.com/): Technical breakdowns of new papers, training techniques, and open-source LLM tooling.
- [Andriy Burkov: The Artificial Intelligence](https://aiweekly.substack.com/): Weekly summary of practical ML and AI news, papers, and engineering insights.
- [Gary Marcus: Marcus on AI](https://garymarcus.substack.com/): Skeptical commentary on AI hype, deep learning's limitations, and policy implications.
- [Andrej Karpathy: Blog](https://karpathy.github.io/): Hands-on essays and tutorials on neural networks, training dynamics, and LLM internals.
- [Lilian Weng: Lil'Log](https://lilianweng.github.io/): In-depth technical posts on RL, LLMs, hallucinations, and agent architectures.
- [Chip Huyen](https://huyenchip.com/blog/): Practical writing on MLOps, AI engineering, and building real-world ML systems.
- [Jack Clark: Import AI](https://importai.substack.com/): Weekly roundup of AI research, policy, and geopolitics.
- [Andrew Ng: The Batch](https://www.deeplearning.ai/the-batch/): Accessible weekly digest of AI news, research, and industry trends.
- [Emily M. Bender & Alex Hanna: Mystery AI Hype Theater 3000](https://buttondown.com/maiht3k): Sharp, linguistically-grounded take-downs of AI hype and overclaims about LLMs.
- [Rachel Thomas: fast.ai blog](https://rachel.fast.ai/): Essays on AI ethics, education, inclusion, and debunking conventional ML wisdom.

### Neural Networks, BERT, attention, Transformers, Word Embeddings, LLMs


- [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/): Foundational textbook covering supervised learning, regularization, trees, and ensembles.
- Van Rijsbergen, C. J. (1979). [Information Retrieval (2nd ed.)](http://www.dcs.gla.ac.uk/Keith/Preface.html). Butterworth-Heinemann.: Classic textbook introducing precision/recall, indexing, and probabilistic retrieval models.
- Wang et al. (2019) [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://aclanthology.org/W18-5446/): Introduces a 9-task benchmark suite designed to evaluate general-purpose language understanding across diverse NLU problems.
- Hu et al. (2020) [XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization](https://dl.acm.org/doi/10.5555/3524938.3525348): Evaluates cross-lingual transfer of multilingual models across 40 languages and 9 tasks.
- Strubell et al. (2019) [Energy and Policy Considerations for Deep Learning in NLP](https://aclanthology.org/P19-1355/): Quantifies the financial and environmental cost of training large NLP models and calls for efficiency-aware research.
- Dodge et al. (2022) [Measuring the Carbon Intensity of AI in Cloud Instances](https://arxiv.org/abs/2206.05229): Measures emissions of cloud-based AI workloads and proposes practices for reducing carbon footprint.
- Sheng et al. (2019) [The Woman Worked as a Babysitter: On Biases in Language Generation](https://aclanthology.org/D19-1339/): Shows that language models generate systematically biased completions across gender, race, and sexual orientation.
- Gupta & Manning (2014) [Improved Pattern Learning for Bootstrapped Entity Extraction](https://aclanthology.org/W14-1611/): Improves bootstrapped entity extraction by jointly learning patterns and entities with better scoring.
- Dou & Neubig (2021) [Word Alignment by Fine-tuning Embeddings on Parallel Corpora](https://aclanthology.org/2021.eacl-main.181/): Uses fine-tuned multilingual embeddings to produce state-of-the-art word alignments without supervision.
- Karpathy, Andrej (2016) [Yes you should understand Backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b): Argues that understanding backpropagation matters because leaky abstractions in autodiff cause real bugs.
- Karpathy, Andrej (2015) [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/): Demonstrates that character-level RNNs can generate surprisingly coherent text across many domains.
- Olah, Christopher (2015) [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/): Visual, intuitive walkthrough of how LSTM gates manage long-range dependencies.
- Olah & Carter (2016) [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/): Surveys attention, neural Turing machines, and other mechanisms that augment RNN capabilities.
- Mikolov et al. (2013) [Efficient Estimation of Word Representations in Vector Space](https://www.semanticscholar.org/paper/Efficient-Estimation-of-Word-Representations-in-Mikolov-Chen/f6b51c8753a871dc94ff32152c00c01e94f90f09): Introduces Word2Vec (CBOW and skip-gram) for learning word embeddings cheaply at scale.
- Pennington et al. (2014) [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/): Learns word vectors by factorizing global word co-occurrence statistics.
- Bojanowski et al. (2017) [Enriching Word Vectors with Subword Information](https://aclanthology.org/Q17-1010/): FastText: represents words as bags of character n-grams to handle morphology and OOV tokens.
- Peters et al. (2018) [Deep Contextualized Word Representations](https://aclanthology.org/N18-1202/): ELMo: produces context-dependent word representations from a deep bidirectional language model.
- Howard & Ruder (2018) [Universal Language Model Fine-tuning for Text Classification](https://aclanthology.org/P18-1031/): ULMFiT: introduces a transfer learning recipe for fine-tuning language models on downstream NLP tasks.
- Devlin et al. (2019) [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding]https://aclanthology.org/N19-1423/): Pre-trains deep bidirectional transformers via masked language modeling, setting a new state of the art on 11 NLP tasks.
- Alammar, Jay (2018) [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/): Visual explanation of how contextual pre-training reshaped NLP transfer learning.
- Vaswani et al. (2017) [Attention Is All You Need](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html): Introduces the Transformer architecture, replacing recurrence with self-attention.
- Uszkoreit, Jakob (2017) [Transformer: A Novel Neural Network Architecture for Language Understanding](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/): Google blog post explaining the intuition behind the Transformer.
- Alammar, Jay (2018) [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): Step-by-step visual breakdown of self-attention and the Transformer encoder/decoder.
- Adaloglou, Nikolas (2020) [How Transformers work in deep learning and NLP: an intuitive introduction](https://theaisummer.com/transformer/): Intuitive introduction to attention, positional encodings, and Transformer mechanics.
- Liu et al. (2019) [RoBERTa: A Robustly Optimized BERT Pretraining Approach](hwww.semanticscholar.org/paper/RoBERTa%3A-A-Robustly-Optimized-BERT-Pretraining-Liu-Ott/077f8329a7b6fa3b7c877a57b81eb6c18b5f87de): Shows BERT was undertrained and yields stronger results with longer training, more data, and no NSP.
- Wolf et al. (2020) [HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://aclanthology.org/2020.emnlp-demos.6/): Presents the open-source library that standardized access to pre-trained transformer models.
- Sun et al. (2019) [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/abs/1905.05583): Empirical study of fine-tuning strategies, learning rates, and layer-wise schedules for BERT.
- Brown et al. (2020) [Language Models are Few-Shot Learners](https://dl.acm.org/doi/abs/10.5555/3495724.3495883): GPT-3: shows that scaling autoregressive LMs to 175B parameters enables strong few-shot in-context learning.
- Gao et al. (2021) [Making Pre-trained Language Models Better Few-shot Learners](https://aclanthology.org/2021.acl-long.295/): LM-BFF: improves few-shot fine-tuning via prompt-based learning and automatic demonstration selection.
- Gao, Tianyu (2021) [Prompting: Better Ways of Using Language Models for NLP Tasks](https://thegradient.pub/prompting/): Survey-style article explaining prompt-based methods and their relationship to fine-tuning.
- Schick & Schütze (2021) [Generating Datasets with Pretrained Language Models](https://www.semanticscholar.org/paper/Generating-Datasets-with-Pretrained-Language-Models-Schick-Sch%C3%BCtze/b769b629c8de35b16735214251d6b4e99cb55762): Uses generative LMs to synthesize labeled training data for sentence-level tasks without human annotation.
- Schick & Schütze (2021) [Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference](https://aclanthology.org/2021.eacl-main.20/): PET: reformulates classification as cloze tasks to leverage MLM knowledge in few-shot settings.
- Bender et al. (2021) [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜](https://dl.acm.org/doi/10.1145/3442188.3445922): Critiques the risks of ever-larger LMs: environmental cost, bias amplification, and illusion of understanding.
- Kirk et al. (2021) [Bias Out-of-the-Box: An Empirical Analysis of Intersectional Occupational Biases in Popular Generative Language Models](https://dl.acm.org/doi/10.5555/3540261.3540461): Audits GPT-2 for occupational stereotypes across intersectional demographic groups.
- Schick et al. (2021) [Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP](https://aclanthology.org/2021.tacl-1.84/): Shows pre-trained LMs can identify and reduce their own biased outputs at decoding time.
- Le Scao et al. (2022) [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://inria.hal.science/hal-03850124/document): Releases an open multilingual LLM trained collaboratively across 46 languages.
- Suau et al. (2022) [Self-conditioning Pre-Trained Language Models](https://proceedings.mlr.press/v162/cuadros22a.html): Identifies expert neurons inside LMs and uses them to control generation without fine-tuning.
- Agüera y Arcas (2022) [Do Large Language Models Understand Us?](https://medium.com/@blaisea/do-large-language-models-understand-us-6f881d6d8e75): Argues that LaMDA-style models exhibit forms of understanding that challenge naive Chinese-room critiques.
- Touvron et al. (2023) [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971): Trains competitive 7B–65B foundation LLMs using only public data with strong inference efficiency.
- Manakul et al. (2023) [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://aclanthology.org/2023.emnlp-main.557/): Detects LLM hallucinations by sampling multiple responses and measuring consistency, no external resources needed.
- Al-Kaswan & Izadi (2023) [The (Ab)use of Open Source Code to Train Large Language Models](https://arxiv.org/abs/2302.13681): Discusses copyright, licensing, and ethical issues of training LLMs on open-source code repositories.
- Luccioni et al. (2024) [Power Hungry Processing: Watts Driving the Cost of AI Deployment?](https://dl.acm.org/doi/10.1145/3630106.3658542): Measures inference-time energy and emissions across tasks, finding generation is far costlier than discriminative tasks.
- Yao et al. (2023) [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629): Interleaves chain-of-thought reasoning with tool-use actions, improving factuality and task success.
- Huyen, Chip (2025) [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/): Practical guide to building applications on top of foundation models, covering evaluation, deployment, and feedback loops.
- Warner et al. (2024) [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Fine Tuning and Inference](https://aclanthology.org/2025.acl-long.127/): ModernBERT: a refreshed encoder with long-context support, faster inference, and stronger downstream performance.
- Chen et al. (2024) [What is the Role of Small Models in the LLM Era: A Survey](https://www.semanticscholar.org/paper/What-is-the-Role-of-Small-Models-in-the-LLM-Era%3A-A-Chen-Varoquaux/e79981c91c1ac40d747377b4af7409793d8e7350/figure/2): Surveys when small models complement or replace LLMs, covering distillation, ensembling, and routing.
- Weng, Lilian (2024) [Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/): Survey blog post on hallucination types, causes, evaluation metrics, and mitigation strategies.
- Mitchell, Melanie (2025) [LLMs and World Models](https://aiguide.substack.com/p/llms-and-world-models-part-1): Examines whether LLMs build genuine world models or rely on shallow heuristics.
- Vafa et al. (2024) [https://dl.acm.org/doi/abs/10.5555/3737916.3738762](https://arxiv.org/abs/2406.03689): Proposes new metrics showing generative models can perform well while harboring incoherent implicit world models.
- Feng et al. (2024) [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201): Revisits minimal LSTMs/GRUs and shows simplified, parallelizable variants rival modern architectures.
- An et al. (2025) [Measuring Gender and Racial Biases in Large Language Models: Intersectional Evidence from Automated Resume Evaluation](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8071848?login=false): Audits LLM-based resume screening and finds intersectional gender × race disparities in hiring recommendations.
- Haim et al. (2025) [What's in a Name? Auditing Large Language Models for Race and Gender Bias](https://arxiv.org/abs/2402.14875): Uses name perturbations to surface racial and gender bias in LLM advice across high-stakes scenarios.
- Bai et al. (2025) [Explicitly Unbiased Large Language Models Still Form Biased Associations](https://www.pnas.org/doi/10.1073/pnas.2416228122): Shows LLMs that pass explicit bias tests still encode stereotypical associations measurable via implicit-association probes.
- Hartzog (2026) [How AI Destroys Institutions](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5181709): Argues AI systems erode the procedural and trust foundations that make institutions work.
