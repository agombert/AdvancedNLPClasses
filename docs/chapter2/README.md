## 🎓 Course Materials

### 📑 Slides

[Download Session 2 Slides (PDF)](../pdfs/2025_BSE_NLP_Session_2.pdf)

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

### 💻 Practical Components

- **Implementing Gradient Descent**: We’ll code a simple neural net from scratch (via NumPy or PyTorch) to see how forward/backward passes work.
- **Vanishing & Exploding Gradients**: In a toy RNN, we’ll visualize how gradients can shrink or explode, and learn about gradient clipping.
- **Recurrent Language Model**: Train an RNN (or LSTM) on a small text corpus (e.g., Tiny Shakespeare) and watch it generate new text sequences.
- **Comparisons**: We’ll show how gating mechanisms (LSTM/GRU) mitigate the issues in vanilla RNNs.
