# Perceptron-Based Logic Circuit in C

Implemented a **bit adder using perceptrons configured as NAND gates**.

The focus is on:
- representing computation as a **graph (adjacency list)**
- executing it as a **DAG**
- ordering execution using **Kahn’s Algorithm**
- handling **manual memory management in C**

---

## Perceptron Configuration (NAND)

Each perceptron is configured to behave as a NAND gate:

- **Weights:** `-2` (for every incoming edge)  
- **Bias:** `+3`

```c
output = (input_sum + bias <= 0) ? 0 : 1;
```
---

## Circuit (Half Adder)

```mermaid
graph LR

A["Input A"]
B["Input B"]

N1["NAND(A,B)"]
N2["NAND(A,N1)"]
N3["NAND(B,N1)"]

SUM["SUM (XOR)"]
CARRY["CARRY (AND)"]

A --> N1
B --> N1

A --> N2
N1 --> N2

B --> N3
N1 --> N3

N2 --> SUM
N3 --> SUM

N1 -->|w1| CARRY
N1 -->|w2| CARRY
```
