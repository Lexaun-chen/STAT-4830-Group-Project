# <center> Optimization in Preference Learning: A Utility-based Probability Prediction Method in Hotel Recommendation System


# Project Overview

When using platforms like **Expedia** to select hotels, users are often presented with a recommendation list tailored to their preferences. A key challenge is determining how to provide the **optimal set of hotel recommendations** from an enormous product pool for each consumer.

![Hotel Recommendation System](https://helios-i.mashable.com/imagery/articles/02Ew6HbGT043olC0GbVwfDe/hero-image.fill.size_1200x900.v1680564973.jpg)
---
### 🎯 Objective
Model **consumer hotel selection behavior** using a **utility-based approach**. Assuming that consumers make choices based on perceived utility, we can estimate their preferences through a **Multinomial Logit Model (MNL)**, where the selection probabilities are driven by utility values and determined via a **SoftMax function**.

### 🚀 Goal
Minimize the error between the **estimated selection probabilities** and the **actual observed probabilities** of a hotel being chosen.

---

### 🔍 Week 3: Simplified Scenario
In the initial simplified scenario, we assume:
- Consumers share **identical preferences**.
- They choose from a **randomized offer set** of 2 to 4 hotels.

This foundational model allows us to establish the basic framework before expanding to more complex cases involving **heterogeneous consumer preferences** and **dynamic choice conditions**.

### 🔍 Week 4&5: Advanced Scenario
Based on the result of initial scenario, we do some extentions :
- Use **real-world data** instead of the synthesis.
- Still assume consumers share **identical preferences**.
- Use **features** to describe hotels and their utilities.
- Consider using new a algorithm ---- **Matrix Completion (SVT)**.

This feature model allows us to further explain the reasons behind consumers' choices. Based on this week's work, we can further use **Multiple Clustering** and then, algorithms like  **Matrix Completion** in future work.
![Week 5 Flowchart](Week%205%20flowchart.png)

<br>

# <center> Week 5 Report
# 1. Problem Statement 

## 1.1 Weekly Overview
This week, we incorporated **item features** into our **regression model** to better specify the utility values of different items. By doing so, we aim to enhance the explanatory power of our model and improve its ability to predict the likelihood of people purchasing items not included in their previous offer set.
Additionally, we utilized a real-world dataset from Expedia, dividing it into training and testing subsets for model evaluation. To analyze the problem, we applied both **linear regression** and a **neural network** framework, comparing their performance and insights.


## 1.2 Background
Expedia offers thousands of hotels, each characterized by attributes such as price, number of rooms, and available space. However, not all hotels are presented to every consumer; we only observe a consumer's choice within their personalized offer set. Our objective is to **model consumer decision-making over the broader universal set of options** and to **predict the likelihood** of a consumer selecting a hotel that was not included in their initial offer set.

To address this, we draw upon the concept of utility maximization, a cornerstone of economic theory. Assuming rational behavior, we posit that consumers evaluate hotels based on their utility, which is determined by the features of each option. By estimating these utility values, we aim to build a model that captures consumer choice behavior more accurately and supports the development of enhanced recommendation systems, thereby improving the match between consumer preferences and offered options.

## 1.3 Data

This week, we utilized a real-world dataset from [Expedia Hotel Business Database](https://www.kaggle.com/datasets/vijeetnigam26/expedia-hotel?resource=download) from 2013, dividing it into training and testing subsets based on time.  Specifically, the **training set** consists of randomly sampled data from January to June, while the **testing set** includes all search records from July onward.

To ensure data quality, we applied the following preprocessing steps:
- **Filtered search groups** to include only those with 15-32 hotel options.
- **Excluded unrealistic entries**, such as prices above $1000 or booking windows over a year.
- **Removed randomly displayed offers** (`random_bool == 0`) to focus on user-driven choices.
- **Ensured at least one booking per search** to capture meaningful decision-making.

Additionally, we **randomly sampled 50,000 training groups and 5,000 test groups** for computational efficiency, resulting in two clean datasets:  
`train_15-32_50000.csv` and `test_15-32_5000.csv`.


# 2. Technical Approach 

## 2.1 Basic Model

Considering the large-scale dataset and the need for time-efficient model training, we utilized the free **T4 GPU** available on Google Colab. 

While both approaches share the same **MNL problem setup**, they differ in the **`optimize_w` function**:
- The first method employs a **traditional optimization approach** using **Adam optimizer** to update the weight matrix `W` directly.
- The second method utilizes a **deep learning-based approach** (`Deep_MNL`), treating `W` as a learnable neural network parameterized by an encoder.

This structural distinction affects how the model updates its parameters during training while maintaining the same loss function and evaluation pipeline.

### **Sales Class**
The `Sales` class handles feature extraction and choice probability calculations. The core function **computes choice probabilities using the softmax function**:

```python
class Sales:
    def __init__(self, all_offer_sets, sell_num, mask):
        self.N_sales = torch.tensor(np.concatenate(sell_num, axis=1), dtype=torch.float64, device=device)
        self.offer_feature = torch.tensor(all_offer_sets, dtype=torch.float64)
        self.original_shape = self.offer_feature.shape
        self.feature_concat = self.offer_feature.reshape(-1, self.offer_feature.shape[2]).to(device)
        self.N = len(all_offer_sets)
        self.mask = torch.tensor(mask, dtype=torch.float64, device=device)
        self.mask_flat = self.mask.reshape((-1,))
        self.masked_feature_concat = self.feature_concat[self.mask_flat == 1]

    def calculate_choice_prob(self, rule_feature, W):
        Z = torch.matmul(rule_feature, W).reshape(self.original_shape[:2])
        masked_Z = Z.masked_fill(self.mask == 0, float('-inf'))
        softmax_result = F.softmax(masked_Z, dim=-1).reshape((-1, 1))[self.mask_flat == 1]  
        return softmax_result  # Selection probabilities of valid options
```
### **MNL Model Definition**

The `Problem_MNL` class defines the MNL model and Computes the Negative Log-Likelihood (NLL) based on the predicted choice probabilities.

```python
class Problem_MNL:
    def __init__(self, S, N, M):
        self.W = None
        self.sales = Sales(S, N, M)
        self.feature_num = S.shape[-1]
        self.sales.feature_concat = self.sales.feature_concat.to(device)

    def loss(self, W):
        N = self.sales.N
        N_sales = self.sales.N_sales
        normalize_term = torch.tensor(1 / N, dtype=torch.float64, device=device)
        fw = self.sales.calculate_all_choice_prob(W)
        fw_log = torch.log(fw)
        return -normalize_term * torch.matmul(N_sales, fw_log)
```
## 2.1 Linear Regression
The Mathematical Formulation:
$$
\sigma(w)_i = \frac{e^{wx_i}}{\sum_{j=1}^{K} e^{wx_j}}
$$ 

represents the probability that a consumer selects hotel 𝑖 from a choice set of 
𝐾 hotels, based on a multinomial logit model.
Here, $𝑥_𝑖$ is a feature vector representing the attributes of hotel 𝑖 (e.g., price, number of rooms, location), and 𝑤 is the weight vector capturing the importance of each feature. The term $𝑤𝑥_𝑖$ represents the utility of hotel 
𝑖, as perceived by the consumer.
```python
def optimize_w(self, lr=5e-6):
    W = torch.zeros(self.feature_num, 1, dtype=torch.float64, requires_grad=True, device=device)
    optimizer = optim.Adam([W], lr=lr)

    for epoch in range(10000):
        LOSS = self.loss(W)
        LOSS.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/10000], Loss: {LOSS.item():.4f}')
```
By incorporating item features into the multinomial logit framework, we compute the probability of each hotel being selected. This approach allows us to model consumer choice behavior more effectively and derive insights into the relative importance of different hotel attributes.

## 2.2 Neural Network 
Next, we consider that the weighted sum of features may fail to capture nonlinear relationships between features and utility values. To address this limitation, we use a neural network to gain deeper insights.

The Deep_MNL neural network calculates utility values through the following structure:

- **Input Layer:**
Accepts a feature vector of a hotel (𝑖𝑛𝑝𝑢𝑡_𝑑𝑖𝑚) that represents attributes such as price, location, and size.

 - **Hidden Layers:**

    Layer 1: A fully connected layer with 100 neurons that applies a linear transformation to the input features. Nonlinearity is introduced using a sigmoid activation function.

    Layer 2: Another fully connected layer with 100 neurons, processing the output of Layer 1 with another linear transformation and sigmoid activation.

- **Output Layer:**

    Layer 3: A fully connected layer with a single neuron, which outputs the predicted utility value for the hotel.

- **Forward Propagation:**

    Data flows sequentially through layer1, layer2, and layer3, with sigmoid activations applied after the first two layers.
```python
def optimize_w(self):
    encoder = Deep_MNL(14).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=5e-5)

    for epoch in range(3000):
        LOSS = self.loss(encoder)
        LOSS.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/3000], Loss: {LOSS.item():.4f}')
```
By replacing the traditional weighted sum with this neural network structure, we can better capture the intricate relationships between hotel features and their utility values, providing a more flexible and accurate framework for modeling consumer choice behavior.

# 3. Initial Results 
We evaluate the **Multinomial Logit (MNL) model** with two training methods:  
(1) **Traditional Optimization** (directly optimizing `W`)  
(2) **Neural Network-based Optimization** (learning `W` using a deep model)

---

## **3.1 Training Performance of Linear MNL**

We trained the MNL model using different learning rates (`1e-5`, `5e-6`, and `1e-6`) while keeping the **Adam optimizer** constant.  
The following figure shows the convergence behavior of **Train and Test Loss**:

![Train and Test Loss](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/tests/Loss%20plot%20in%20Linear%20MNL.png?raw=true)
### **Key Observations:**
- A higher learning rate (`1e-5`) leads to **faster convergence**, but may risk **overshooting**.
- A lower learning rate (`1e-6`) results in **slow and stable** convergence.
- Train and Test Loss remain closely aligned, suggesting **no overfitting**.

---

### **Feature Importance Analysis:**

After training, we analyze the learned feature weights to interpret **what factors influence user choices the most**.

![Feature Importance](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/tests/Feature%20Importance%20in%20Linear%20MNL.png?raw=true)
### **Insights:**
- **`price_usd` has a strong negative effect**, indicating price sensitivity.
- **Hotel-related features (e.g., `prop_starrating`) are highly weighted**, suggesting users value hotel attributes.
- **`Position` (ranking in search results) also influences user choice**, the negative weight meaning sonsumers prefer higher ranks .

---

## **3.2 Training Performance of NN-based MNL**

The model is trained for **3000 iterations**, optimizing the **Negative Log-Likelihood (NLL) loss** using the **Adam optimizer**.  
The following plot illustrates the loss trend over training iterations:

![NN MNL Loss](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/tests/Loss%20plot%20in%20nn%20MNL.png?raw=true)

### **Key Observations:**
- The **train and test loss decrease steadily**, indicating stable learning.
- Final Loss Values:
  - **Train Loss:** `2.8916`
  - **Test Loss:** `2.9055`
- The test loss is slightly higher than the train loss, suggesting **good generalization**.

---

## **3.3 Summary: Linear vs. Neural Network-Based MNL**
| Model | Training Strategy | Convergence Speed | Final Train Loss | Final Test Loss | Generalization |
|--------|------------------|------------------|----------------|----------------|----------------|
| **Linear MNL** | Directly optimizes `W` | Slower | 3.048 | 3.058 | Moderate |
| **NN-Based MNL** | Learns `W` via `Deep_MNL` | Faster | **2.8916** | **2.9055** | Better |
 
- **NN-based MNL achieves a lower final loss** than the traditional approach.
- **Neural networks optimize faster**, reducing training time while achieving **better generalization**.
- Future work includes **hyperparameter tuning** and exploring **alternative deep architectures** for further improvements.

---


# 4. Next Steps 

## 4.1 Future Direction
In future work, we plan to incorporate **matrix completion** into our approach. Given the high dimensionality of hotel features, running a multinomial logit (MNL) model with all features can be **computationally expensive**. 
To address this, we propose leveraging **unsupervised learning techniques** to efficiently **classify both hotels and consumers into distinct clusters**. By grouping similar entities, we can reduce the complexity of the problem while preserving meaningful patterns in consumer choice behavior. 
Matrix completion can then be applied within these clustered datasets to infer missing preferences and improve recommendation accuracy, making the model both scalable and robust.

## 4.2 Detailed Pathways
To implement this approach, we will follow these key steps:

1. **Refined Data Cleaning & Aggregation**  
   - Only retain users with **more than one purchase record**.  
   - Aggregate multiple purchases by the same user into a **unified consumption profile**.

2. **User & Hotel Classification**  
   - **Hotel Classification:** Perform **multiple clustering** based on the two most significant features identified in the **linear regression model**:  
     - **Price (`price_usd`)**  
     - **Search Rank Position (`position`)**  
   - **User Classification:** Categorize consumers based on the **types of hotels they have previously chosen**.

3. **Matrix Completion using Singular Value Thresholding (SVT)**  
   - We are currently implementing the **SVT algorithm** to serve as the primary method for **matrix completion**.  
   - The goal is to reconstruct missing preference data efficiently, allowing better **recommendation modeling**.

---

## 4.3 Key Technical Challenges

- **Limited Applicability for New Consumers:**  
  - The proposed approach is most effective for consumers with **multiple consumption records**.  
  - **Challenge:** It is difficult to predict the **likelihood distribution** of **new consumers** who have no prior purchases.  

- **Computational Bottlenecks in Matrix Completion:**  
  - Our current **SVT algorithm** achieves an **error level of `1e-4`**, indicating promising accuracy.  
  - However, two major issues remain:
    1. **High Rank of the Reconstructed Matrix:** The recovered matrix **exhibits high rank**, which limits its interpretability and effectiveness.  
    2. **Slow Execution Speed:** The algorithm's **runtime is currently inefficient**, making large-scale application impractical.

To overcome these challenges, we will explore:
- **Alternative regularization techniques** to enforce a **low-rank structure**.  
- **Optimized SVT implementations** to enhance computational efficiency.  
- **Hybrid models** integrating deep learning to improve performance.  

---
## 5. Group members
| Name          | Email                  |
|:-------------:|:----------------------:|
| Shuhan Zhang  | zhang19@sas.upenn.edu  |
| Xinyu Zhang   | joyxyz@sas.upenn.edu   |
| Lexuan Chen   | amy0305@sas.upenn.edu  |
