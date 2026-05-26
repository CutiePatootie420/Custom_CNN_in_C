![CNN Illustration](./images/cnn%20illustration.png)
# MNIST digit classifier from scratch in C

A lightweight, high-performance 4-layer Multilayer Perceptron (MLP) written entirely from scratch in single-threaded C. This engine contains **zero external machine learning libraries or frameworks**—only standard C headers (`stdlib.h`, `math.h`, `stdio.h`, etc.) are used. 

The network achieves **91.74% test accuracy** on the 10,000-image MNIST test set in just 5 training epochs.

---

## Key Features
* **Zero Dependencies:** Built entirely from scratch in C.
* **Backpropagation:** Full analytical gradient derivation across all hidden layers.
* **Softmax + Cross-Entropy:** Mathematically decoupled backpropagation for high stability and fast learning.
* **Xavier-Glorot approximated initialization:** Healthy weight variance to prevent vanishing or exploding gradients.
* **100% Memory & Resource Safe:** Zero memory leaks, proper cleanup, and closed file descriptors.
* **Fast CPU Training:** Custom progress-logging limits IO bottlenecks, executing thousands of mini-batch updates in minutes on a single CPU thread.

---

## Network Architecture
The network uses a compressed architecture to balance performance and capacity:

* **Input Layer (L0):** 784 Neurons (MNIST raw pixels, normalized $[0, 1]$)
* **Hidden Layer 1 (L1):** 16 Neurons (Sigmoid activation)
* **Hidden Layer 2 (L2):** 16 Neurons (Sigmoid activation)
* **Output Layer (L3):** 10 Neurons (Softmax activation + Cross-Entropy Loss)
* Note: No biases are allocated for the input layer (L0).

---

## Deep Learning Mathematics 

### 1. Softmax Activation (Probability Outputs)
In the final layer, we map the raw outputs $z^{(3)}_i$ to a probability distribution where all activations sum to $1$:

$$a^{(3)}_i = \frac{e^{z^{(3)}_i}}{\sum_k e^{z^{(3)}_k}}$$

#### 💡 Numerical Stability (Softmax Normalization)
Exponentiating large raw outputs ($z_i \geq 80$) leads to floating-point overflows (`inf`), which infects the training gradients with `NaN`. To prevent this, this engine implements **Softmax Normalization** by subtracting the maximum raw value in the output layer ($z_{\text{max}}$) before exponentiating:

$$a^{(3)}_i = \frac{e^{z^{(3)}_i - z_{\text{max}}}}{\sum_k e^{z^{(3)}_k - z_{\text{max}}}}$$

This is mathematically identical but bounds all exponent inputs to $\leq 0$, ensuring $e^{\text{negative}} \leq 1.0$, completely protecting the network from runtime crashes.

---

### 2. The Softmax + Cross-Entropy Mathematical Miracle
When you train a network using **Mean Squared Error (MSE)** and Sigmoid outputs, the learning rates drop to near-zero as the network gets confident because the derivative of Sigmoid $\sigma'(z)$ saturates.

By combining **Softmax** activation with **Cross-Entropy Loss**:
$$\mathcal{L} = -\sum_i y_i \ln(a_i)$$

the derivative of the loss with respect to the output raw weighted input $z_i$ simplifies to:

$$\frac{\partial \mathcal{L}}{\partial z_i} = a_i - y_i$$

This is a mathematical miracle because the derivative of the Softmax quotient rule and the derivative of the natural log **perfectly cancel out all divisions, logs, and exponents!** 

In C, the error is simply `cost.parameter[i] = activation[i] - target[i]`. This removes all complex output-layer calculations from the backpropagation chain, making updates extremely fast and stable.

---

### 3. Xavier (Glorot) Weight Initialization
Weights are initialized randomly using a uniform distribution in the range `[-0.5, 0.5]`. 

This range approximates the mathematically ideal **Xavier Initialization** standard deviation ($\sigma = \sqrt{\frac{2}{N_{in} + N_{out}}}$) for the hidden layers:
* **L1 $\rightarrow$ L2:** Ideal range is approx `[-0.43, 0.43]`
* **L2 $\rightarrow$ L3:** Ideal range is approx `[-0.48, 0.48]`

By initializing near these exact ranges, the network ensures that signal variance remains constant across all layers, preventing early vanishing or exploding gradients.

---

## Systems

* **Preventing IO Bottlenecks:** Logging is throttled to print only once every 500 batches, eliminating terminal output latency and speeding up training by over 20x.
* **Leak-Free Memory Safety:** Implements strict, reversed-order heap cleanup (`free`) for all dynamically allocated matrices and safe closure of IDX binary file handles (`fclose`) upon termination.

---

## How to Compile and Run

1. Make sure you have the MNIST dataset binary files inside a directory named `mnist/` inside the project root:
   * `train-images-idx3-ubyte`
   * `train-labels-idx1-ubyte`
   * `t10k-images-idx3-ubyte`
   * `t10k-labels-idx1-ubyte`

2. Compile the source code using `gcc` (with `-O3` optimization for automatic vectorization):
   > gcc -Wall -O3 mnist_digit_cnn.c -o mnist_digit_cnn -lm

3. Run the binary:
   > ./mnist_digit_cnn

---

## Sample Output
```text
Epoch 0, mini-batch 0, iteration 31:
Prediction: 7 | Actual: 6

Epoch 2, mini-batch 500, iteration 31:
Prediction: 1 | Actual: 1

Epoch 4, mini-batch 1500, iteration 31:
0: 0.006124
1: 0.005128
2: 0.039874
3: 0.010214
4: 0.123024
5: 0.008793
6: 0.037789
7: 0.595895
8: 0.011101
9: 0.161070
Prediction: 7 | Actual: 7

=========================================
FINAL TEST SET ACCURACY: 91.74% (9174/10000 correct)
=========================================



