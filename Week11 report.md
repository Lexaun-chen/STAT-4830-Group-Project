# Matrix Factorization with Noisy SGD and Momentum/EMA Enhancements

## Motivation

Matrix completion is a core component of modern recommender systems. In real-world scenarios like Netflix or Amazon, the user-item rating matrix is highly sparse. To recover missing entries, low-rank matrix factorization is a widely adopted method.

In this report, we explore three strategies to improve the stochastic gradient descent (SGD) optimization used for matrix factorization:

- **Vanilla SGD**
- **SGD with Momentum**
- **SGD with Exponential Moving Average (EMA)**

We introduce noise into the updates (Noisy SGD) and analyze the effect of momentum and EMA on convergence and reconstruction accuracy.

## Problem Setup


We consider a partially observed matrix $$\( X \in \mathbb{R}^{m \times n} \)$$, where some entries are missing (`NaN`).  
Let the observed index set be:

$$
\Omega = \{(i, j) \mid X_{ij} \text{ is observed}\}
$$

**Goal**: Recover missing entries of \( X \) by finding a low-rank matrix approximation.

---
 ### Low-Rank Matrix Factorization

Assume $$\( X \)$$ is approximately low-rank. We aim to find:

$$
X \approx \hat{X} = U V^\top
$$

- $$\( U \in \mathbb{R}^{m \times k} \)$$: latent features of users  
- $$\( V \in \mathbb{R}^{n \times k} \)$$: latent features of items  
- $$\( k \ll \min(m, n) \)$$: low-rank dimension

To improve expressiveness, we add **bias terms**:

$$
X_{ij} \\approx \\hat{X}_{ij} = b + b_i + b_j + U_i^\top V_j
$$

- $$\( b \)$$: global average bias  
- $$\( b_i \), \( b_j \)$$: user/item biases

Only observed entries (non-missing values) in \\(X\\) are used to optimize the loss function.

### Loss Function (Squared Error + Regularization)

We minimize the error only on observed entries:

$$
\mathcal{L} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} (X_{ij} - \hat{X}_{ij})^2 + \beta (\|U\|_F^2 + \|V\|_F^2)
$$

- $$\( \| \cdot \|_F \$$): Frobenius norm  
- $$\( \beta \)$$: regularization coefficient

---

## Training Strategies

### 1. Vanilla SGD

Each iteration updates parameters using the gradient of the observed error:

$$
\\begin{aligned}
e_{ij} &= X_{ij} - \hat{X}_{ij} 
\end{aligned}
$$

$$
\begin{aligned}
U_i &\leftarrow U_i + \alpha (2e_{ij} V_j - \beta U_i) \\
V_j &\leftarrow V_j + \alpha (2e_{ij} U_i - \beta V_j)
\end{aligned}
$$
$$
\\begin{aligned}
b_i &\leftarrow b_i + \alpha (2 e_{ij} - \beta b_i) \\
b_j &\leftarrow b_j + \alpha (2 e_{ij} - \beta b_j)
\end{aligned}
$$

### 2. SGD with Momentum

Momentum accumulates past gradients to accelerate convergence:

$$
\\begin{aligned}
\\Delta U_i &= \\gamma \\Delta U_i + \\alpha (2e_{ij} V_j - \\beta U_i) \\\\
\\Delta V_j &= \\gamma \\Delta V_j + \\alpha (2e_{ij} U_i - \\beta V_j)
\\end{aligned}
$$

Then:

$$
U_i \\leftarrow U_i + \\Delta U_i \\quad \\text{and} \\quad V_j \\leftarrow V_j + \\Delta V_j
$$

### 3. SGD with EMA

Instead of accumulating momentum, we use exponentially smoothed gradients:

$$
EMA_{U_i} &\leftarrow \beta \cdot EMA_{U_i} + (1 - \beta) \cdot \text{grad}_{U_i} 
$$

$$
EMA_{V_j} &\leftarrow \beta \cdot EMA_{V_j} + (1 - \beta) \cdot \text{grad}_{V_j}
$$

Then update:

$$
U_i \leftarrow U_i + EMA_{U_i}, \quad V_j \leftarrow V_j + EMA_{V_j}
$$

To stabilize EMA updates, we apply **gradient and parameter clipping**.

### 4. Adam Optimizer (from PyTorch)

PyTorch's `Adam` optimizer combines momentum and adaptive learning rate:

$$
\theta_t \leftarrow \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Where:
- $$\( \hat{m}_t \)$$: momentum (1st moment)
- $$\( \hat{v}_t \)$$: RMS of gradients (2nd moment)

---


## Experiments

### 1. Small-Scale Evaluation (10x10, Rank = 2)

| Strategy | Final MSE   | Notes                 |
| -------- | ----------- | --------------------- |
| Vanilla  | ~0.028      | Slow but steady       |
| Momentum | ~0.011      | Fast convergence      |
| EMA      | **~0.003**  | Very quick early drop |

➡️ **Observation**: On smaller matrices, EMA helps accelerate learning in the early stages.

![Small Matrix Vanilla SGD Loss Curve](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/tests/Basic%20SGD%20Error%20Small%20Scale.png.png?raw=true)
![Small Matrix NQM Loss](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/tests/MSE%20Small%20Scale.png.png?raw=true)
![Small Matrix NQM Error](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/tests/Error%20Small%20Scale.png.png?raw=true)

---

### 2. Large-Scale Evaluation (500x500, Rank = 10)

| Strategy | Final MSE   | Notes                                 |
| -------- | ----------- | ------------------------------------- |
| Vanilla  | ~0.038      | Slower convergence                    |
| Momentum | **~0.007**  | Best performance for large-scale      |
| EMA      | ~0.015      | Stable after clipping, but fluctuates |

➡️ **Observation**: For larger matrices, **Momentum** outperforms others in terms of convergence speed and final accuracy.

![tests/Basic SGD Error Large Scale.png.png](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/tests/Basic%20SGD%20Error%20Large%20Scale.png.png?raw=true)
![tests/MSE Large Scale.png.png](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/tests/MSE%20Large%20Scale.png.png?raw=true)
![tests/Error Large Scale.png.png](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/tests/Error%20Large%20Scale.png.png?raw=true)

---

### Training Curves Summary

- Momentum improves convergence steadily over time.
- EMA shows fast early reduction but stabilizes at a higher plateau.
- Vanilla remains the most conservative strategy.

## Conclusion

We introduced noise-aware matrix factorization methods with both **momentum** and **EMA** enhancement strategies. From empirical results:

- EMA helps in **early training** or **small data**.
- Momentum offers better **long-term stability** and is more suited for **large-scale recommendation** tasks.

## Next Steps

- Apply these methods to real recommendation datasets (e.g., MovieLens)
- Tune per-strategy hyperparameters for optimal performance
- Explore adaptive EMA schedules (warm-up or decay)

---

**Notebook**: [Colab Link](https://colab.research.google.com/drive/1AZWJ9Za3fyonyTNOJA_B9gg6jDu9Bi81)

