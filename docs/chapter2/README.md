## 🎓 Course Materials

### 📑 Slides

[Download Session 2 Slides (PDF)](../pdfs/BSE_NLP_Session_2.pdf)

### 📓 Notebooks

  - [Intro to Neural Nets & Backprop (NumPy)](Session_2_1_NeuralNets_with_Numpy.ipynb)
  - [Simple RNN for Text Generation (Tiny Shakespeare)](Session_2_2_Text_Generation_with_RNN.ipynb)
  - [LSTM for Sequence Classification](Session_2_3_LSTM_Classif.ipynb)

---

## 🚀 Session 2: Neural Networks, Backpropagation & RNNs

In this second session, we move beyond the baselines of Session 1 and dive into the **world of neural networks**. From the foundational **vanilla feedforward** architecture to more advanced **recurrent neural networks**, you’ll see how we capture sequential patterns crucial for language understanding. We’ll also explore the intricacies of training these models, including **gradient descent** variants and **backpropagation**, as well as potential pitfalls like **vanishing** or **exploding gradients**.

### 🎯 Learning Objectives

1. **Understand** the core mechanics of neural networks, from feedforward passes to computing gradients.
2. Master **backpropagation** to see how weight updates flow through each layer.
3. Explore **Recurrent Neural Networks (RNNs)** and see why they’re pivotal for handling sequential data such as text.
4. Learn about **Long Short-Term Memory (LSTM)** networks and how they solve the shortcomings of vanilla RNNs.
5. Build a **text generator** that can produce plausible sequences, using an RNN trained on a small dataset (Tiny Shakespeare).

### 📚 Topics Covered

#### Neural Network Essentials

- **Vanilla Networks**: Single-layer networks, the chain rule in practice, and how we compute partial derivatives for each parameter.
- **Gradient Descent**: A closer look at **batch**, **mini-batch**, and **stochastic** variants. We’ll discuss how they’re used in frameworks like PyTorch or TensorFlow.

#### Recurrent Neural Networks (RNNs)

- **Sequential Data**: Why standard NNs fail to capture dependencies in text, time-series, or speech.
- **Vanishing/Exploding Gradients**: Common training challenges in RNNs and strategies to mitigate them.
- **Practical RNN Implementations**: Adopting RNN variants, including LSTM and GRU, for tasks like language modeling and sequence labeling.

### 📖 Bibliography & Recommended Reading

- **Karpahty A. (2016) "Yes You Should Understand Backprop"** - [Blog Post](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)
  A blog post explaining backpropagation in detail.

- **Silviu P. (2016) "Written Memories: Understanding, Deriving and Extending the LSTM"** - [Blog Post](https://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm)
  A blog post explaining RNN and LSTM logic in detail.

- **Colah, C. (2015) "Understanding LSTMs"** - [Blog Post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  A blog post explaining LSTMs in detail.

- **Karpathy, A. (2015) "The Unreasonable Effectiveness of Recurrent Neural Networks"** - [Blog Post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  Classic blog post illustrating RNN text generation (tiny Shakespeare).

- **Colah, C. (2016) "Attention and Augmented Recurrent Neural Networks"** - [Blog Post](https://distill.pub/2016/augmented-rnns/)
  A blog post explaining augmented RNNs in detail.

- **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning internal representations by error propagation."**  [Paper](https://ieeexplore.ieee.org/document/6302929)
  Presents the backpropagation algorithm in detail.

- **Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory."**  [Paper](https://ieeexplore.ieee.org/abstract/document/6795963)
  Neural Computation, 9(8).
  Original LSTM paper addressing the vanishing gradient problem in RNNs.

- **Cho et al. (2014). "Learning phrase representations using RNN encoder-decoder for statistical machine translation."**  [Paper](https://aclanthology.org/D14-1179.pdf)
  Introduced the GRU (Gated Recurrent Unit) as a simpler alternative to LSTM.

- **He et al. (2015). "Deep Residual Learning for Image Recognition."**  [Paper](https://ieeexplore.ieee.org/document/7780459)
  Not directly NLP, but the notion of “degradation problem” is generalizable to deep networks.

- **LeCun et al. (1989). "Backpropagation Applied to Handwritten Zip Code Recognition."**  [Paper](https://ieeexplore.ieee.org/document/6795724)
  Foundational convolutional neural network trained end-to-end with backpropagation.

- **Ruder, S. (2016). "An Overview of Gradient Descent Optimization Algorithms."**  [Paper](https://arxiv.org/abs/1609.04747)
  Survey of SGD variants (Momentum, Nesterov, Adagrad, RMSprop, Adam) used to train neural nets.

- **Qian, N. (1999). "On the Momentum Term in Gradient Descent Learning Algorithms."**  [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608098001166)
  Introduces the momentum term that accelerates gradient descent.

- **Nesterov, Y. (1983). "A Method for Solving the Convex Programming Problem with Convergence Rate O(1/k²)."**  [Paper](https://www.mathnet.ru/eng/dan46009)
  Nesterov accelerated gradient, a look-ahead variant of momentum.

- **Duchi et al. (2011). "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization."**  [Paper](https://jmlr.org/papers/v12/duchi11a.html)
  Adagrad: per-parameter adaptive learning rates.

- **Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization."**  [Paper](https://arxiv.org/abs/1412.6980)
  The Adam optimizer combining momentum and adaptive learning rates.

- **Glorot & Bengio (2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks."**  [Paper](https://proceedings.mlr.press/v9/glorot10a.html)
  Xavier/Glorot weight initialization for stable deep-network training.

- **Pascanu et al. (2013). "On the Difficulty of Training Recurrent Neural Networks."**  [Paper](https://proceedings.mlr.press/v28/pascanu13.html)
  Analyzes vanishing/exploding gradients and proposes gradient clipping.

- **Graves, A. (2013). "Generating Sequences With Recurrent Neural Networks."**  [Paper](https://arxiv.org/abs/1308.0850)
  LSTM-based sequence generation, including peephole connections.

### 💻 Practical Components

- **Implementing Gradient Descent**: We’ll code a simple neural net from scratch (via NumPy or PyTorch) to see how forward/backward passes work.
- **Vanishing & Exploding Gradients**: In a toy RNN, we’ll visualize how gradients can shrink or explode, and learn about gradient clipping.
- **Recurrent Language Model**: Train an RNN (or LSTM) on a small text corpus (e.g., Tiny Shakespeare) and watch it generate new text sequences.
