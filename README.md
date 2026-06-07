# Neural Network from Scratch in C

A high-performance, multi-threaded neural network library written entirely in C — **zero ML frameworks, zero external libraries**. Only standard C and POSIX headers.

Trains on the full 60,000-image MNIST handwritten digit dataset and achieves **91.84% test accuracy** in **2.7 seconds** on an 8-core Apple Silicon CPU.

![CNN Illustration](./images/cnn%20illustration.png)

---

## Performance

The final architecture is **100x faster** than the original single-threaded baseline, verified across identical hyperparameters (5 epochs, batch size 32, learning rate 0.1):

| Version | Training Time | Speedup |
|---|---|---|
| V1 — Single-threaded, pointer-chased `double**` matrices | 270 s | 1x |
| V2 — Contiguous 1D arrays, recursive backprop, 8-core pthreads, SIMD auto-vectorization | **2.7 s** | **100x** |

### Where the 100x Comes From

| Optimization | What Changed |
|---|---|
| **Algorithmic** | Replaced redundant $O(N^3)$ output-layer backprop with recursive $\delta$-based $O(N^2)$ propagation |
| **Memory** | Replaced scattered `double**` pointer tables with flat contiguous 1D arrays — unlocks L1/L2 cache prefetching |
| **Parallelism** | Distributed mini-batches across all 8 CPU cores via POSIX Threads (`pthreads`) with thread-local gradient buffers |
| **SIMD** | Contiguous memory layout allows the compiler to auto-vectorize inner loops (verified with `-Rpass=loop-vectorize`) |

---

## Architecture

The network architecture is fully dynamic — any number of hidden layers with any number of nodes. The default configuration used for benchmarks:

```
Input (784) → Hidden 1 (16, Sigmoid) → Hidden 2 (16, Sigmoid) → Output (10, Softmax + Cross-Entropy)
```

### Memory Layout

All network parameters are stored in flat 1D arrays indexed via precomputed prefix sums (`p_sums`) and weight offset tables (`w_indices`). This eliminates pointer chasing and enables the CPU to stream data sequentially through cache lines.

```
biases:   [ --- L0 (784, unused) --- | --- L1 --- | --- L2 --- | ... | --- Ln --- ]
                                       ^                         ^
                                  p_sums[1]                 p_sums[n]

weights:  [ --- L0→L1 --- | --- L1→L2 --- | ... | --- L(n-1)→Ln --- ]
                ^                                       ^
           w_indices[0]                            w_indices[n-1]
```

### Multi-Threading Model

Each thread in the pool receives a partition of the mini-batch and computes forward pass + backpropagation independently into **thread-local gradient buffers** (`dL_dw`, `dL_db`). After all threads finish (`pthread_join`), the main thread aggregates gradients and updates the shared weights. This design has **zero race conditions** — threads only read from shared weights during the forward pass and write exclusively to their own local buffers.

---

## Deep Learning Mathematics

### Softmax + Cross-Entropy Loss

The output layer uses Softmax activation:

$$a^{(L)}_i = \frac{e^{z^{(L)}_i}}{\sum_k e^{z^{(L)}_k}}$$

Combined with Cross-Entropy loss $\mathcal{L} = -\sum_i y_i \ln(a_i)$, the gradient simplifies to:

$$\frac{\partial \mathcal{L}}{\partial z^{(L)}_i} = a^{(L)}_i - y_i$$

This eliminates all log/exp/division from the backward pass. In C: `dL_dz[i] = activation[i] - target[i]`.

### Recursive Backpropagation

Hidden layer error signals are computed recursively, propagating backward from the output:

$$\delta^{(l)}_i = \left(\sum_j \delta^{(l+1)}_j \cdot w^{(l)}_{ij}\right) \cdot a^{(l)}_i \cdot (1 - a^{(l)}_i)$$

This avoids redundantly recomputing the entire forward chain for every layer. Weight gradients are then:

$$\frac{\partial \mathcal{L}}{\partial w^{(l)}_{ij}} = \delta^{(l+1)}_j \cdot a^{(l)}_i$$

### Xavier-Glorot Weight Initialization

Weights are initialized from a uniform distribution in `[-0.5, 0.5]`, which closely approximates the ideal Xavier standard deviation $\sigma = \sqrt{\frac{2}{N_{in} + N_{out}}}$ for the hidden layer configurations used, preventing vanishing/exploding gradients during early training.

---

## File Structure

| File | Purpose |
|---|---|
| `main.c` | Training loop, thread orchestration, gradient aggregation, evaluation |
| `mlp.c` / `mlp.h` | Network creation, memory allocation, weight initialization, cleanup |
| `thread_handler.c` / `thread_handler.h` | Thread state management, forward pass, backpropagation |
| `file_handler.c` / `file_handler.h` | MNIST IDX binary file parser (big-endian) |

---

## Build & Run

### Prerequisites
- GCC or Clang with C99 support
- MNIST dataset files in a `mnist/` directory:
  - `train-images-idx3-ubyte`
  - `train-labels-idx1-ubyte`
  - `t10k-images-idx3-ubyte`
  - `t10k-labels-idx1-ubyte`

> MNIST can be downloaded from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).

### Compile

```bash
gcc -Wall -O3 main.c mlp.c file_handler.c thread_handler.c -o main -lm
```

To verify SIMD auto-vectorization:
```bash
gcc -Wall -O3 -Rpass=loop-vectorize main.c mlp.c file_handler.c thread_handler.c -o main -lm
```

### Run

```bash
./main
```

The program will prompt you for:
1. Number of hidden layers
2. Nodes per hidden layer
3. Learning rate
4. Batch size
5. Number of epochs

### Sample Output

```
Enter the number of hidden layers you want in your MLP for MNIST dataset: 2
Enter number of nodes in hidden layer 0: 16
Enter number of nodes in hidden layer 1: 16
Enter desired learning rate: 0.1
Enter desired bacth size (preferred like 2,4,16,32...): 32
Enter desired epochs: 5
avl cores: 8
Starting training...
Training Time (New Parallelized): 2.692629 seconds
Training finished
Evaluating on test set...
91.840000 accuracy on 5 epochs, 32 batch size, 0.100000 learning rate
```

---

## Built With

`stdlib.h` · `stdio.h` · `math.h` · `string.h` · `time.h` · `pthread.h` · `unistd.h`

Nothing else.
