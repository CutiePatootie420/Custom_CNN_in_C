# Neural Network from Scratch in C

A high-performance, multi-threaded neural network library written entirely in C — **zero ML frameworks, zero external libraries**. Only standard C and POSIX headers.

Trains on the full 60,000-image MNIST handwritten digit dataset and achieves **91.84% test accuracy** in **2.7 seconds** on an 8-core Apple Silicon CPU.

![CNN Illustration](./images/cnn%20illustration.png)

---

## Performance

The repository contains two versions:
1. **V1 (Legacy Baseline)**: A single-threaded, matrix-pointer-chased (`double**`) implementation with a **hardcoded architecture** (fixed layers, nodes, and hyperparameters).
2. **V2 (Optimized)**: A modular, cache-friendly, multi-threaded (`pthreads`) implementation with a **fully dynamic architecture** (allows user-defined layers, nodes, learning rates, batch sizes, and epochs at runtime).

The optimized implementation is **100x faster** than the original baseline, verified across identical hyperparameters (5 epochs, batch size 32, learning rate 0.1):

| Version | Source Code | Architecture Type | Training Time | Speedup |
|---|---|---|---|---|
| **V1 (Baseline)** | `mnist_digit_cnn.c` | **Hardcoded** (784 → 16 → 16 → 10) | 270 s | 1x |
| **V2 (Optimized)** | `main.c` (+ modular files) | **Fully Dynamic** (User-defined at runtime) | **2.7 s** | **100x** |

### Where the 100x Comes From

| Optimization | What Changed |
|---|---|
| **Algorithmic** | Replaced redundant $O(N^3)$ output-layer backprop with recursive $\delta$-based $O(N^2)$ propagation |
| **Memory** | Replaced scattered `double**` pointer tables with flat contiguous 1D arrays — unlocks L1/L2 cache prefetching |
| **Parallelism** | Distributed mini-batches across all 8 CPU cores via POSIX Threads (`pthreads`) with thread-local gradient buffers |
| **SIMD** | Contiguous memory layout allows the compiler to auto-vectorize inner loops (verified with `-Rpass=loop-vectorize`) |

---

## Architecture

While the legacy version (**V1**) uses a hardcoded structure, the new optimized version (**V2**) is **fully dynamic** — supporting any number of hidden layers with any number of nodes per layer. 

The default configuration used for benchmarks matching the baseline's topology:

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

### SIMD Auto-Vectorization & Cache Coherency

By structuring all weights, biases, and activations into flat 1D contiguous arrays, the CPU can stream data sequentially through cache lines. This layout eliminates pointer chasing and enables the compiler (using `-O3`) to automatically vectorize the inner loops using hardware SIMD registers (NEON on ARM/Apple Silicon, AVX on x86).

Compiling with `-Rpass=loop-vectorize` confirms active auto-vectorization of critical loops:
* Feedforward dot-product loops are vectorized (width 2, interleave 4).
* Activation load loops are vectorized (width 16, interleave 1).
* Dynamic weight gradient accumulation loops are vectorized (width 2, interleave 4).

### Dynamic Multi-Threading Model

Thread initialization and creation are kept **fully dynamic**. At runtime, the program queries the operating system for the number of available CPU cores (using `sysconf(_SC_NPROCESSORS_ONLN)` on UNIX/macOS and `GetSystemInfo` on Windows) and spawns exactly `min(available_cores, batch_size)` threads to maximize hardware utilization.

Each worker thread receives a partition of the mini-batch and computes the forward pass + backpropagation independently. Because threads write exclusively to their own **thread-local gradient buffers** (`dL_dw`, `dL_db`) and only read from the shared weights during forward propagation, the system achieves **zero race conditions** and avoids all lock/mutex synchronization overhead. The main thread aggregates the gradients and updates the shared network parameters only after all worker threads finish execution (`pthread_join`).

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

The project code is divided into legacy baseline and optimized components:

### Legacy Implementation (V1)
* `mnist_digit_cnn.c` — Single-file, single-threaded pointer-chased matrix implementation. Uses a **hardcoded** architecture configuration.

### Optimized Modular Implementation (V2)
| File | Purpose |
|---|---|
| `main.c` | Training loop, thread orchestration, gradient aggregation, evaluation. Runs the **fully dynamic** setup |
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

To compile the optimized dynamic multi-threaded version:
```bash
gcc -Wall -O3 main.c mlp.c file_handler.c thread_handler.c -o main -lm
```

To verify SIMD auto-vectorization:
```bash
gcc -Wall -O3 -Rpass=loop-vectorize main.c mlp.c file_handler.c thread_handler.c -o main -lm
```

To compile the legacy hardcoded version:
```bash
gcc -Wall -O3 mnist_digit_cnn.c -o mnist_digit_cnn -lm
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
