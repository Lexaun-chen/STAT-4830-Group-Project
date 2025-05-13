# STAT-4830-Group-Project
## Group Members: Shuhan Zhang, Xinyu Zhang, Lexuan Chen
## Summary
This project addresses the problem of predicting consumer hotel preferences based on personalized offer sets and relevant hotel and consumer features, using real-world data from Expedia. Two modeling frameworks were explored: a mixture preference learning model optimized via the Frank-Wolfe algorithm, including neural network extensions, and a matrix completion approach based on low-rank factorization methods. The findings indicate that linear models generally outperform neural networks due to limited data size and inherent linearity in consumer decision-making. Additionally, the matrix completion methods proved robust and effective, particularly when incorporating bias-aware initialization and using the Huber loss function to handle sparsity and noisy data. Overall, the study provides valuable insights into modeling consumer preferences and demonstrates practical methods suitable for predicting choices in large-scale recommendation systems.
## Repository Structure Overview
## Setup Instructions
## Running the Code
## Executable Demo Link
# Frank-Wolfe Algorithm (MNL and NN-based)
https://colab.research.google.com/drive/1u8P8WUMnW8a9PnYXDeWyapwQbmQUbVnf?usp=sharing
# Matrix Completion Algorithm (NN-based+Adam)
Netflix datadet:https://colab.research.google.com/drive/1ZZg2OyqT9Gd4FOCcCQV-kTcAWMcw1YDI#scrollTo=_qPSiFMlWu-S
Expedia dataset: https://colab.research.google.com/drive/10Qhc49LnFdxzMEYaxjaSKuz9-rmb6DoU
