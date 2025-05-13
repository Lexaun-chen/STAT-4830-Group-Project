# STAT-4830-Group-Project
## Group Members: Shuhan Zhang, Xinyu Zhang, Lexuan Chen

## Summary
This project addresses the problem of predicting consumer hotel preferences based on personalized offer sets and relevant hotel and consumer features, using real-world data from Expedia. Two modeling frameworks were explored: a mixture preference learning model optimized via the Frank-Wolfe algorithm, including neural network extensions, and a matrix completion approach based on low-rank factorization methods. The findings indicate that linear models generally outperform neural networks due to limited data size and inherent linearity in consumer decision-making. Additionally, the matrix completion methods proved robust and effective, particularly when incorporating bias-aware initialization and using the Huber loss function to handle sparsity and noisy data. Overall, the study provides valuable insights into modeling consumer preferences and demonstrates practical methods suitable for predicting choices in large-scale recommendation systems.

## Repository Structure Overview

## Setup Instructions

## Running the Code
The entire final results can be executed seamlessly in Google Colab with one click.
Our final notebook automatically imports all core functions from the src/ directory and performs end-to-end training, evaluation, and visualization. Unit tests from the tests/ folder can also be executed within the notebook. 
To run the project: 👉 Click “Runtime → Run all” in Colab — everything will install and execute automatically, no manual setup required.
For more details and a breakdown of each step, see the Colab demo section.

## Executable Demo Link

## Frank-Wolfe Algorithm (MNL and NN-based)
https://colab.research.google.com/drive/1u8P8WUMnW8a9PnYXDeWyapwQbmQUbVnf?usp=sharing

## Matrix Completion Algorithm (NN-based+Adam)

### Netflix datadet
https://colab.research.google.com/drive/10wc_ViKCkdIKrYfXoaxzNkpGt_y-oBNq?usp=sharing

#### Netflix Matrix Completion Workflow
This section outlines the full matrix completion pipeline for the Netflix dataset using low-rank factorization and Adam optimization. The goal is to recover a dense user–movie rating matrix from sparse observed entries.
1️⃣ Load Netflix Ratings Data
We first load the user–movie ratings matrix using the raw `.dat` file from the MovieLens-style Netflix dataset. Each entry contains:
- `UserID`, `MovieID`, `Rating`, `Timestamp`
The ratings are stored as a sparse matrix with shape `[num_users, num_movies]`, where missing entries are marked as `NaN`.

2️⃣ Construct Sparse Utility Matrix
We transform the raw ratings list into a matrix format, mapping UserID and MovieID to contiguous indices. This results in a sparse 2D PyTorch tensor X, where:
	•	Rows represent users
	•	Columns represent movies
	•	Values are ratings (1–5), missing entries as NaN

3️⃣ Train Matrix Completion Model with Adam
We randomly mask a portion of the observed ratings as test data using create_train_test_split, and train the matrix completion model to reconstruct the observed entries using Adam optimizer.

4️⃣ Predict Full Utility Matrix
After training, we extract the completed matrix with all missing entries filled. This matrix can be used for recommendation, evaluation, or ranking.


### Expedia dataset
https://colab.research.google.com/drive/1JG8rfqfAN9tAbaT9fhMazpSH6vbZX72Z?usp=sharing

#### Expedia Matrix Completion Workflow
This section describes the full pipeline we implemented to recover a dense user–item utility matrix for Expedia hotel recommendation, combining preference learning with matrix completion.

 1️⃣ Train MNL Model on Expedia Raw Data
We first train a Multinomial Logit (MNL)-style model using user choice logs from Expedia. Each user's offer set is represented as a list of hotel feature vectors, and a 3-layer neural network (`NeuralNet3Layer`) is trained to predict utility scores under a utility-based selection rule.

- **Input**: Offer set features (price, star rating, location, etc.) per session  
- **Output**: Choice probabilities (softmax over utilities)  
- **Training Target**: Selected hotel (index of 1 in offer set)  
- **Loss**: Cross-entropy over choice probability (via the `Sales` class)

2️⃣ Generate Utility Scores for New Samples
Once the MNL model is trained, we use it to score a new batch of Expedia data (e.g., validation/test users) by computing each user’s predicted utility for every hotel in their offer set.
This produces a sparse utility matrix, where each row corresponds to a user session, and non-missing entries are the predicted utilities from the MNL model.

3️⃣ Complete Sparse Utility Matrix with Adam Optimizer
The sparse utility matrix is then passed into our ExpediaMatrixCompletion class, which applies low-rank matrix factorization (with bias terms) to fill in the missing values.
The completion model is trained using an Adam optimizer to minimize masked MSE/Huber loss on observed entries, with L2 regularization on latent factors.
After training, the final output is a dense matrix of predicted utilities for all user–item pairs, suitable for downstream ranking or recommendation tasks:
```python
# Retrieve the completed matrix
X_full = completion_model.filled_matrix()
```
