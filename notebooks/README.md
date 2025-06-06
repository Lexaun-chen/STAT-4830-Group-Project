# Demo Summary
## Frank-Wolfe Algorithm Demo Workflow
This section outlines the full pipeline for preference learning on the Expedia dataset using a mixture model optimized via the Frank-Wolfe algorithm. The objective is to predict consumer choices by identifying latent preference types and their proportions from personalized hotel offers.
#### 1️⃣ Load Expedia Dataset
We first load the Expedia dataset containing personalized hotel offers. The data includes tensors representing:
offerset_tensor: Features of offered hotels
sell_tensor: Actual consumer choices (binary indicators)
mask_tensor: Binary masks indicating available hotel options per offer set
#### 2️⃣ Initialize Preference Learning Model
We encapsulate the data into a Sales class and initialize a Problem_FrankWolfe object. This model represents:
Offers (hotel features)
Consumers' choice data
Latent preference components, represented by utility vectors to be discovered through optimization
#### 3️⃣ Train Model with Frank-Wolfe Algorithm
We iteratively apply the Frank-Wolfe optimization procedure, alternating between identifying new latent preference types (support finding) by solving linear minimization problems and updating their mixture proportions with Adam or L-BFGS optimizers. The goal is to minimize the Negative Log-Likelihood (NLL) between predicted and actual choices.
#### 4️⃣ Predict Consumer Choice Probabilities
Upon convergence, we obtain a learned mixture model comprising utility vectors and their associated proportions. This completed model can then predict consumer choices across new offer sets, enabling better recommendations and insights into consumer preferences.

## Netflix Matrix Completion Demo Workflow
This section outlines the full matrix completion pipeline for the Netflix dataset using low-rank factorization and Adam optimization. The goal is to recover a dense user–movie rating matrix from sparse observed entries.
#### 1️⃣ Load Netflix Ratings Data
We first load the user–movie ratings matrix using the raw `.dat` file from the MovieLens-style Netflix dataset. Each entry contains:
- `UserID`, `MovieID`, `Rating`, `Timestamp`
The ratings are stored as a sparse matrix with shape `[num_users, num_movies]`, where missing entries are marked as `NaN`.

#### 2️⃣ Construct Sparse Utility Matrix
We transform the raw ratings list into a matrix format, mapping UserID and MovieID to contiguous indices. This results in a sparse 2D PyTorch tensor X, where:
	•	Rows represent users
	•	Columns represent movies
	•	Values are ratings (1–5), missing entries as NaN

#### 3️⃣ Train Matrix Completion Model with Adam
We randomly mask a portion of the observed ratings as test data using create_train_test_split, and train the matrix completion model to reconstruct the observed entries using Adam optimizer.

#### 4️⃣ Predict Full Utility Matrix
After training, we extract the completed matrix with all missing entries filled. This matrix can be used for recommendation, evaluation, or ranking.


## Expedia Matrix Completion Demo Workflow
This section describes the full pipeline we implemented to recover a dense user–item utility matrix for Expedia hotel recommendation, combining preference learning with matrix completion.

 #### 1️⃣ Train MNL Model on Expedia Raw Data
We first train a Multinomial Logit (MNL)-style model using user choice logs from Expedia. Each user's offer set is represented as a list of hotel feature vectors, and a 3-layer neural network (`NeuralNet3Layer`) is trained to predict utility scores under a utility-based selection rule.

- **Input**: Offer set features (price, star rating, location, etc.) per session  
- **Output**: Choice probabilities (softmax over utilities)  
- **Training Target**: Selected hotel (index of 1 in offer set)  
- **Loss**: Cross-entropy over choice probability (via the `Sales` class)

#### 2️⃣ Generate Utility Scores for New Samples
Once the MNL model is trained, we use it to score a new batch of Expedia data (e.g., validation/test users) by computing each user’s predicted utility for every hotel in their offer set.
This produces a sparse utility matrix, where each row corresponds to a user session, and non-missing entries are the predicted utilities from the MNL model.

#### 3️⃣ Complete Sparse Utility Matrix with Adam Optimizer
The sparse utility matrix is then passed into our ExpediaMatrixCompletion class, which applies low-rank matrix factorization (with bias terms) to fill in the missing values.
The completion model is trained using an Adam optimizer to minimize masked MSE/Huber loss on observed entries, with L2 regularization on latent factors.
After training, the final output is a dense matrix of predicted utilities for all user–item pairs, suitable for downstream ranking or recommendation tasks:
```python
# Retrieve the completed matrix
X_full = completion_model.filled_matrix()
```
