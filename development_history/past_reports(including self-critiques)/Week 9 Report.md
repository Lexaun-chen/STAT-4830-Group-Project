# <center> Optimization in Preference Learning: A Utility-based Probability Prediction Method in Hotel Recommendation System

# <center> Week 9 Report
# 1. Problem Statement 

# Frank-Wolfe Framework for Consumer Preference Modeling

## 1. Overview

### 1.1 Background
Consumer choice modeling is an essential task in economics and marketing, aiming to predict the likelihood of different consumer groups selecting specific products. One common approach is the Multinomial Logit (MNL) model, which estimates consumer preferences using observed choices. However, traditional MNL models assume a homogeneous consumer population, which may not accurately represent market diversity. To address this, we implement a **Frank-Wolfe optimization framework** to iteratively refine consumer preference models by introducing multiple consumer types and optimizing their proportions.

### 1.2 Data
The dataset consists of:
- **Product feature matrix (`p_offerset`)**: A set of product characteristics used to train consumer preference models.
- **Sales data (`p_sell`)**: Observed purchase frequencies for different products, used as training labels.
- **Masking matrix (`p_mask`)**: A binary mask indicating product availability for different consumers.

## 2. Technical Approach

### 2.1  Model Architecture
The core model, `Preference`, is a simple neural network implemented as a PyTorch `nn.Module`:

Input: Offer set features (14 dimensions per product).
Layer: A single linear layer (nn.Linear) mapping features to a scalar utility value.
Output: Log-probabilities of choice via F.log_softmax, masked to exclude padded entries.
The Sales class stores tensorized offer sets, sales data, and masks, providing key metadata (e.g., number of offers, products, features).

### 2.2 Frank-Wolfe Optimization
- Equation
$$
P_{ij} = \sum_{m=1}^{M} \alpha_m \cdot P_{ij}^{(m)}
$$

- $M$: Number of consumer types (set to 50 in the code)  
- $\alpha_m$: Proportion of the $m$-th consumer type  
- $P_{ij}^{(m)}$: Choice probability for the $m$-th consumer type:

  $$
  P_{ij}^{(m)} = \frac{e^{u_{ij}^{(m)}}}{\sum_{k \in S_i} e^{u_{ik}^{(m)}}}
  $$

- $u_{ij}^{(m)} = \mathbf{w}_m \cdot \mathbf{x}_{ij} + b_m$: Utility for the $m$-th consumer type

- Initialization:
Trains an initial MNL model using Adam optimizer and Negative Log-Likelihood (NLL) loss over 300 epochs.
Stores the initial preference model and choice probabilities.
- Support Finding:
Identifies new consumer types by optimizing a new Preference instance to maximize alignment with the gradient of the current NLL loss.
Uses Adam optimizer over 500 epochs, minimizing a custom loss (support_finding_loss).
- Proportion Update:
Adjusts the mixture proportions of consumer types by optimizing a softmax-weighted NLL loss over 3000 epochs.
Updates the main problem loss and gradient after each iteration.
The train_frank_wolfe function executes this process for a specified number of consumer types (50 in the code), tracking NLL loss per iteration.

### 2.3 Training Configuration
Device: CPU (GPU checked but unavailable in the provided output).
Batch Size: 1024.
Learning Rates: `5e-3` (initialization/proportion update), `5e-2` (support finding).
Epochs: `300` (initial), `500` (support finding), `3000` (proportion update).

## 3. Initial Results
- The initial training phase successfully fits an MNL model to the observed sales data.
- The **Frank-Wolfe iterations** progressively introduce additional consumer types, improving model flexibility.
- The **proportion update step** dynamically adjusts the importance of different consumer segments.
- Preliminary evaluations show a reduction in **negative log-likelihood**, indicating improved model performance.


- Initial NLL Loss: Approximately `2.95` (after `300` epochs).
- Iterative Improvement: Each iteration refines the mixture model, reducing NLL loss (specific values truncated in the output but trend downward, e.g., ~`2.85` after one iteration).
- Consumer Types: Up to 50 distinct preference profiles, each with associated proportions.

![Week 9 Result](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/development_history/past_reports(including%20self-critiques)/Week%209%20-result.png?raw=true)


## 4. Next steps
To advance the current model, we propose:
- Inter-Item Effects: Investigate how items in an offer set influence each other (e.g., substitution effects) by adding interaction terms to the utility function and analyzing choice correlations.
- Matrix Completion: Apply low-rank matrix factorization to handle sparse data, imputing missing choices to enhance robustness and initialize consumer types.
- Validation: Integrate these enhancements, validate on test data, and assess improvements in NLL loss and accuracy.
These steps aim to improve prediction and uncover deeper choice dynamics efficiently.





## 5. Group members
| Name          | Email                  |
|:-------------:|:----------------------:|
| Shuhan Zhang  | zhang19@sas.upenn.edu  |
| Xinyu Zhang   | joyxyz@sas.upenn.edu   |
| Lexuan Chen   | amy0305@sas.upenn.edu  |
