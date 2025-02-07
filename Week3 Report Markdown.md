

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
We employ [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) as the optimizer due to its efficiency in handling sparse data. The model is implemented in [PyTorch](https://pytorch.org/) , following a structured pipeline:

```
Model Construction → Data Import → Training → Testing & Validation
```
Performance is assessed using **RMSE, Accuracy**, and **NLL**, comparing predicted probabilities against actual choices. We utilize the [Expedia Hotel Business Database](https://www.kaggle.com/datasets/vijeetnigam26/expedia-hotel/data) from Kaggle, but challenges include dataset biases and limited visibility into individual user histories. Our approach prioritizes scalability and interpretability, ensuring robust consumer preference modeling.

---
## 3. Initial Results


The model demonstrates **stable convergence**, with both original and frequency-domain losses plateauing at **0.3581** and **0.1289**, respectively, maintaining consistency beyond **Epoch 1000**. The alignment between **in-sample** (original: 0.358, frequency: 0.129) and **out-of-sample** metrics (original: 0.355, frequency: 0.125) indicates **robust generalization performance**. 

However, training progress appears to **halt prematurely**, suggesting potential inefficiencies related to **optimizer settings** or **model capacity constraints**.

---

### 3.1 Model Limitations & Future Directions

Our current research is conducted within a **highly simplified experimental framework**, leading to a model that lacks sufficient:
- **Feature complexity**
- **Explanatory depth**
- **Predictive robustness**

While the model shows **marginal performance gains** over naive baselines (e.g., simple mean estimation), its **limited utility for real-world applications** is evident.

---

### 3.2 **Proposed Enhancements**

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

### 3.3 **Key Takeaways**

- **Stable convergence** observed, but with signs of early training stagnation.
- **Model performance** remains close to naive baselines, highlighting the need for enriched features.
- Future work will focus on **feature complexity** and **consumer segmentation** to enhance both **robustness** and **real-world applicability**.

---
## 4. Next Steps

Our current model assumes homogeneous consumer preferences, though real-world data suggests
multiple preference types. A study by Jagabathula et al. (2020) explores choice modeling under
heterogeneous consumer behaviors, but its methodology lacks a Python implementation. We aim to
bridge this gap by developing a tailored solution. However, several technical challenges remain,
including selecting an appropriate solver, improving our proficiency with PyTorch, managing large-
scale datasets within GPU constraints, and parallelizing computations for varying choice sets. To
refine our approach, we seek further guidance on PyTorch best practices, access to relevant project
examples, and alternative modeling strategies for customer segmentation. While no alternative
approaches have been identified yet, continued exploration is necessary. Thus we have established a
structured framework for optimization and gained insights into formulating the problem efficiently.

---
## 5. Group members
| Name          | Email                  |
|:-------------:|:----------------------:|
| Shuhan Zhang  | zhang19@sas.upenn.edu  |
| Xinyu Zhang   | joyxyz@sas.upenn.edu   |
| Lexuan Chen   | amy0305@sas.upenn.edu  |
