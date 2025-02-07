

# Optimization in Utility-Based Hotel Recommendation System
---

## 1. Problem Statement

---
## 1.1 Overview

When using platforms like **Expedia** to select hotels, users are often presented with a recommendation list tailored to their preferences. A key challenge is determining how to provide the **optimal set of hotel recommendations** from an enormous product pool for each consumer.

### 🎯 Objective
Model **consumer hotel selection behavior** using a **utility-based approach**. Assuming that consumers make choices based on perceived utility, we can estimate their preferences through a **Multinomial Logit Model (MNL)**, where the selection probabilities are driven by utility values and determined via a **SoftMax function**.

### 🔍 Simplified Scenario
In the initial simplified scenario, we assume:
- Consumers share **identical preferences**.
- They choose from a **randomized offer set** of 2 to 4 hotels.

This foundational model allows us to establish the basic framework before expanding to more complex cases involving **heterogeneous consumer preferences** and **dynamic choice conditions**.

### 🚀 Goal
Minimize the error between the **estimated selection probabilities** and the **actual observed probabilities** of a hotel being chosen.

---

![Hotel Recommendation System](https://helios-i.mashable.com/imagery/articles/02Ew6HbGT043olC0GbVwfDe/hero-image.fill.size_1200x900.v1680564973.jpg)

---
## 1.2 Business Significance

While individual choices may vary, consumers tend to exhibit a **consistent probability distribution** when presented with the same set of options. Understanding this decision-making process allows platforms like **Expedia** to **optimize recommendation algorithms** and ultimately **enhance booking conversion rates**.

---

## 1.3 Measurement Methods

- **Root Mean Square Error (RMSE)**  
- **Accuracy**  
- **Negative Log-Likelihood (NLL)**  

These metrics are used to evaluate the performance of our model in estimating consumer choice probabilities.

---

## 1.4 Data & Utility

Despite the potential benefits, several challenges arise:  
- **Limited visibility** into the exact number of hotels viewed by each user.  
- **Variability** in the offer sets presented across different sessions.  
- The impracticality of defining a **true probability distribution** for every possible combination of hotel sets.

To address these issues, we focus on **inferring the utility of each hotel** rather than attempting to model the true probability distribution directly. This approach offers a **scalable** and **interpretable** solution, enabling robust insights into consumer preferences.

Our analysis relies on **Expedia user booking data** and **hotel attributes**, but **data limitations**—such as incomplete visibility of the offer sets shown—add complexity to the modeling process. By framing the problem around **utility estimation** instead of direct probability prediction, we aim to develop a model that is both **robust** and **adaptable** to varying conditions, effectively capturing consumer decision-making patterns.

---

## 2. Technical Approach
---
### 2.1  Mathematical Formulation 

We model consumer hotel selection using a **Multinomial  Logit (MNL) Model** , where the probability of choosing hotel from a set is:
$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$ with _i_ representing the latent utility of the hotel. This framework builds on the intuition introduced by [Batsell and Polking (1985)](https://doi.org/10.1287/mksc.4.3.177), who proposed a market share model based on choice probabilities and competitive interactions. 


Our objective is to minimize prediction error using **Negative Log- Likelihood (NLL)** as the loss function:
$$
L = - \sum_{n} \sum_{i \in S_n} y_{n,i} \log P_{n,i}
$$ 

---
### 2.2 PyTorch Implementation & Validation Methods

```python
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

# Basic Model
class DiagonalLinear(nn.Module):
    def __init__(self, size):
        super(DiagonalLinear, self).__init__()
        self.size = size
        self.diag = nn.Parameter(torch.ones(size))  # Initialize diagonal entries as ones
    def forward(self, x):
        # Construct diagonal matrix
        diag_matrix = torch.diag(self.diag)
        return x @ diag_matrix


class MNL(nn.Module):
    def __init__(self, OptSize):
        super().__init__()
        self.input_linear = DiagonalLinear(OptSize)

    def forward(self, e):
        mask = e == 1
        e = self.input_linear(e)
        masked_e = torch.where(mask == 1, e, float('-inf'))
        masked_softmax_e = F.softmax(masked_e, dim=-1)
        return masked_softmax_e, e
```
We employ [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) as the optimizer due to its efficiency in handling sparse data. The model is implemented in [PyTorch](https://pytorch.org/) , following a structured pipeline:

```
Model Construction → Data Import → Training → Testing & Validation
```
Performance is assessed using **RMSE, Accuracy**, and **NLL**, comparing predicted probabilities against actual choices. We utilize the [Expedia Hotel Business Database](https://www.kaggle.com/datasets/vijeetnigam26/expedia-hotel/data) from Kaggle, but challenges include dataset biases and limited visibility into individual user histories. Our approach prioritizes scalability and interpretability, ensuring robust consumer preference modeling.

---
## 3. Initial Results
---
## 3.1 Two Synthetic Choice Data Tests
#### Test 1 --- Probability
 We first test the situation with homogenous pre-defined probability distribution.
 ```python
 def generate_one_hot(probabilities):
    probabilities = np.array(probabilities)
    p_index = np.random.choice(len(probabilities), p=probabilities)
    one_hot = np.zeros_like(probabilities)
    one_hot[p_index] = 1
    return one_hot

product_set = [0, 1, 2, 3]
offer_set = [0, 1, 2, 3]

hypothetical_choice_p = [[0.98, 0.02, 0, 0],
                         [0.5, 0, 0.5, 0],
                         [0.5, 0, 0, 0.5],
                         [0, 0.5, 0.5, 0],
                         [0, 0.5, 0, 0.5],
                         [0, 0, 0.9, 0.1],
                         [0.49, 0.01, 0.5, 0],
                         [0.49, 0.01, 0, 0.5],
                         [0.5, 0, 0.45, 0.05],
                         [0, 0.5, 0.45, 0.05],
                         [0.49, 0.01, 0.45, 0.05]]
 ```

```bash
#Training Results
Epoch [5/100],   Original Loss: 0.3781, Frequency Loss: 0.1770
Epoch [10/100],  Original Loss: 0.3738, Frequency Loss: 0.1677
Epoch [15/100],  Original Loss: 0.3701, Frequency Loss: 0.1593
Epoch [20/100],  Original Loss: 0.3670, Frequency Loss: 0.1520
Epoch [25/100],  Original Loss: 0.3646, Frequency Loss: 0.1460
Epoch [30/100],  Original Loss: 0.3627, Frequency Loss: 0.1411
Epoch [35/100],  Original Loss: 0.3612, Frequency Loss: 0.1374
Epoch [40/100],  Original Loss: 0.3602, Frequency Loss: 0.1346
Epoch [45/100],  Original Loss: 0.3594, Frequency Loss: 0.1327
Epoch [50/100],  Original Loss: 0.3589, Frequency Loss: 0.1313
Epoch [55/100],  Original Loss: 0.3586, Frequency Loss: 0.1304
Epoch [60/100],  Original Loss: 0.3584, Frequency Loss: 0.1298
Epoch [65/100],  Original Loss: 0.3583, Frequency Loss: 0.1294
Epoch [70/100],  Original Loss: 0.3582, Frequency Loss: 0.1292
Epoch [75/100],  Original Loss: 0.3581, Frequency Loss: 0.1291
Epoch [80/100],  Original Loss: 0.3581, Frequency Loss: 0.1290
Epoch [85/100],  Original Loss: 0.3581, Frequency Loss: 0.1289
Epoch [90/100],  Original Loss: 0.3581, Frequency Loss: 0.1289
Epoch [95/100],  Original Loss: 0.3581, Frequency Loss: 0.1289
Epoch [100/100], Original Loss: 0.3581, Frequency Loss: 0.1289
```
**Performance Metrics**

| **Metric**               | **In-Sample** | **Out-of-Sample** |
|--------------------------|---------------|-------------------|
| **Original RMSE**        | 0.3581        | 0.3605            |
| **Frequency RMSE**       | 0.1289        | 0.1418            |

![Training Loss Curve](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/tests/Test%20result%20with%20pre-defined%20probability%20distribution.png?raw=true)
The model demonstrates **stable convergence**, with both original and frequency-domain losses plateauing at **0.3581** and **0.1289**, respectively, maintaining consistency beyond **Epoch 1000**. The alignment between **in-sample** (original: 0.358, frequency: 0.129) and **out-of-sample** metrics (original: 0.355, frequency: 0.125) indicates **robust generalization performance**. 

However, training progress appears to **halt prematurely**, suggesting potential inefficiencies related to **optimizer settings** or **model capacity constraints**.

---

### Test 2 --- Utility

In the second test, we predefined the ground truth utilities for each alternative and added Gumbel noise to simulate individual choice randomness. Synthetic choice data was generated based on these noisy utilities, and an MNL model was trained to recover the underlying utility parameters. The results show that while the captured utilities differ in absolute values, the model successfully learned the correct preference ranking, validating its effectiveness in choice modeling.

```python
def generate_synthetic_choice_data(num_alternatives, num_observations, utilities):

    choice_data = np.zeros((num_observations, num_alternatives))

    for i in range(num_observations):
        # Add random noise to utilities
        noisy_utilities = utilities + np.random.gumbel(size=num_alternatives)

        # Choose the alternative with the highest utility
        choice = np.argmax(noisy_utilities)

        # Update choice data
        choice_data[i, choice] = 1

    return choice_data
     
num_alternatives = 4
num_observations = 10000
ground_truth_utilities = np.array([0.5, 0.3, 0.2, 0.1])
choice_data = generate_synthetic_choice_data(num_alternatives, num_observations, ground_truth_utilities)
```

- The training loss decreased from **0.1873** to **0.1859**, stabilizing after ~230 epochs, indicating quick convergence.
- **Ground Truth Utilities:** `[0.5, 0.3, 0.2, 0.1]`
- **Captured Utilities:** `[1.1956, 1.0035, 0.8680, 0.7821]`

While the captured utilities are scaled differently from the ground truth, the **preference ranking is preserved**, confirming that the MNL model effectively learned the underlying choice behavior. This aligns with the MNL property where relative differences in utilities drive the choice probabilities.

---

### 3.2 Model Limitations & Future Directions

Our current research is conducted within a **highly simplified experimental framework**, leading to a model that lacks sufficient:
- **Feature complexity**
- **Explanatory depth**
- **Predictive robustness**

While the model shows **marginal performance gains** over naive baselines (e.g., simple mean estimation), its **limited utility for real-world applications** is evident.

---

### 3.3 **Proposed Enhancements**

To address these limitations, we propose the following refinements:
1. **Incorporating Item-Specific Attributes:**  
   Integrate features such as:
   - **Temporal** (e.g., seasonality effects)
   - **Spatial** (e.g., location-based factors)
   - **Socioeconomic indicators**
2. **Introducing Consumer Segmentation Strategies:**  
   Explicitly model **heterogeneous preference patterns** by differentiating between user subgroups, such as:
   - **Price-sensitive consumers**
   - **Quality-driven consumers**

These enhancements aim to:
- **Improve interpretability**
- **Increase estimation accuracy**
- Better align the framework with the **complex dynamics of real-world decision-making** processes.

---

### 3.4 **Key Takeaways**

- **Stable convergence** observed, but with signs of early training stagnation.
- **Model performance** remains close to naive baselines, highlighting the need for enriched features.
- Future work will focus on **feature complexity** and **consumer segmentation** to enhance both **robustness** and **real-world applicability**.

---
## 4. Next Steps
---
### 4.1 Detailed Pathways

Building on the two enhancement directions outlined in **Section 3.3**, we have identified practical pathways to improve our model’s performance and adaptability.

####  - Integrating Item-Specific Attributes

To incorporate item-specific features, we will employ two complementary approaches:
- **Linear Regression:** To capture simple, interpretable relationships between item attributes and choice probabilities.
- **Neural Networks:** To model complex, non-linear interactions that may not be easily captured through linear methods.

Both methods will be evaluated to determine their effectiveness in enhancing model accuracy and interpretability.


#### - Modeling Heterogeneous Consumer Preferences

Our current model assumes **homogeneous consumer preferences**, which oversimplifies real-world decision-making where diverse consumer segments exist. To address this:
- We will explicitly model **heterogeneous preference patterns** by differentiating between consumer subgroups (e.g., price-sensitive vs. quality-driven consumers).
- Drawing inspiration from [Jagabathula et al. (2020)](https://orcid.org/0000-0002-4854-3181), which explores choice modeling under heterogeneous behaviors, we aim to develop a **Python-based implementation** to bridge the gap left by existing methodologies.

---

### 4.2. Key Technical Challenges

While refining our model, we anticipate the following technical hurdles:
- **Solver Selection:** Identifying an efficient and scalable solver tailored to our problem structure.
- **PyTorch Proficiency:** Enhancing our expertise with PyTorch to optimize model performance and manage complex architectures.
- **Large-Scale Data Management:** Addressing GPU memory constraints when handling large datasets, including strategies for **efficient parallelization** across varying choice sets.

---

### 4.3 Future Exploration

To refine our approach, we seek:
- **Guidance on PyTorch Best Practices:** Techniques for optimizing model performance and handling large-scale data efficiently.
- **Access to Relevant Case Studies:** Practical examples of projects with similar objectives to inform our methodology.
- **Alternative Modeling Strategies:** Exploration of advanced segmentation methods to capture nuanced consumer behaviors.

Although we have not yet identified alternative approaches, our current framework provides a solid foundation for continuous optimization. We are committed to ongoing exploration to enhance both the robustness and interpretability of our model.

---
## 5. Group members
| Name          | Email                  |
|:-------------:|:----------------------:|
| Shuhan Zhang  | zhang19@sas.upenn.edu  |
| Xinyu Zhang   | joyxyz@sas.upenn.edu   |
| Lexuan Chen   | amy0305@sas.upenn.edu  |
