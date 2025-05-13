# src/adam_netflix_completion.py

"""
Neural network and matrix factorization utilities for matrix completion (e.g., on the Netflix dataset).
Includes:
    - MatrixFactorization (embedding-based)
    - Netflix_MF_Adam (bias-aware matrix completion)
    - Training and evaluation functions
    - Data loading and preprocessing tools

Designed for use in a structured project under `src/`.
"""

import math
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.nn.utils import clip_grad_norm_

# =========================
# Matrix Factorization (Embedding-based)
# =========================
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=30):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_indices, item_indices):
        user_vecs = self.user_embedding(user_indices)
        item_vecs = self.item_embedding(item_indices)
        return (user_vecs * item_vecs).sum(dim=1)

# =========================
# Matrix Completion with Bias Terms
# =========================
class Netflix_MF_Adam(nn.Module):
    def __init__(self, X_train, train_mask, k):
        super().__init__()
        self.device = X_train.device
        self.X = X_train
        self.k = k
        self.m, self.n = X_train.shape
        self.mask = train_mask
        self.n_obs = self.mask.sum().item()
        scale = 1.0 / math.sqrt(k)

        self.U = nn.Parameter(torch.randn(self.m, k, device=self.device) * 0.02)
        self.V = nn.Parameter(torch.randn(self.n, k, device=self.device) * 0.02)

        mean_value = torch.nanmean(X_train).item()
        self.b = nn.Parameter(torch.tensor(mean_value, dtype=torch.float32, device=self.device))
        user_means = torch.nanmean(X_train, dim=1)
        item_means = torch.nanmean(X_train, dim=0)
        self.b_u = nn.Parameter((user_means - mean_value).clone().detach())
        self.b_v = nn.Parameter((item_means - mean_value).clone().detach())

    def forward(self):
        pred = self.b + self.b_u[:, None] + self.b_v[None, :] + self.U @ self.V.T
        return pred

    def train_loss(self, loss_type="mse", l2_lambda=1e-4, huber_delta=1.0):
        pred = self.forward()
        true = self.X[self.mask]
        pred = pred[self.mask]

        if loss_type == "mse":
            loss = F.mse_loss(pred, true)
        elif loss_type == "mae":
            loss = F.l1_loss(pred, true)
        elif loss_type == "huber":
            loss = F.huber_loss(pred, true, delta=huber_delta)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        reg = l2_lambda * (self.U.norm() ** 2 + self.V.norm() ** 2) / self.n_obs
        return loss + reg

    def full_matrix(self):
        return self.forward().detach()

    def evaluate_train_rmse(self):
        pred = self.forward()[self.mask]
        true = self.X[self.mask]
        return F.mse_loss(pred, true).sqrt().item()

    def filled_matrix(self, pred_matrix=None):
        if pred_matrix is None:
            pred_matrix = self.forward().detach()
        filled = self.X.clone().detach()
        filled[~self.mask] = pred_matrix[~self.mask]
        return filled

# =========================
# Training Function
# =========================
def train_model(model, train_data, epochs=10, lr=0.01, weight_decay=1e-5, clip_norm=5.0):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    user_tensor = torch.LongTensor(train_data[:, 0])
    item_tensor = torch.LongTensor(train_data[:, 1])
    rating_tensor = torch.FloatTensor(train_data[:, 2])

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(user_tensor, item_tensor)
        loss = F.mse_loss(preds, rating_tensor)
        loss.backward()
        clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# =========================
# Evaluation Function
# =========================
def evaluate_model(model, test_data):
    model.eval()
    user_tensor = torch.LongTensor(test_data[:, 0])
    item_tensor = torch.LongTensor(test_data[:, 1])
    true_ratings = test_data[:, 2]

    with torch.no_grad():
        preds = model(user_tensor, item_tensor).numpy()

    rmse = mean_squared_error(true_ratings, preds, squared=False)
    mae = mean_absolute_error(true_ratings, preds)

    return {"RMSE": rmse, "MAE": mae}

# =========================
# Seed Setter
# =========================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Data Loaders
# =========================
def load_netflix_data(ratings_url):
    response = requests.get(ratings_url)
    lines = response.text.strip().split('\n')
    ratings_list = [line.split("::") for line in lines]
    df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp']).astype(int)
    return df[['UserID', 'MovieID', 'Rating']].values

def load_movies_data(movies_url):
    response = requests.get(movies_url)
    lines = response.text.strip().split('\n')
    film_list = [line.split("::") for line in lines]
    df = pd.DataFrame(film_list, columns=['MovieID', 'Title', 'Genres'])
    df['MovieID'] = df['MovieID'].astype(int)
    return df

# =========================
# Custom Train/Test Split for Matrix Completion
# =========================
def create_train_test_split(X, num_test=2500, seed=123):
    """
    Split a matrix X into training and test sets by masking out a subset of observed entries.

    Args:
        X (torch.Tensor): Rating matrix with NaN for missing entries.
        num_test (int): Number of entries to mask out for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        X_train (torch.Tensor): Training matrix with test entries masked as NaN.
        train_mask (torch.BoolTensor): Mask indicating observed training entries.
        test_idx (torch.LongTensor): Indices of held-out test entries.
        test_ratings (torch.Tensor): True values of test entries.
    """
    observed_idx = torch.nonzero(~torch.isnan(X), as_tuple=False)
    perm = torch.randperm(observed_idx.size(0), generator=torch.Generator().manual_seed(seed))
    test_idx = observed_idx[perm[:num_test]]

    X_train = X.clone()
    X_train[test_idx[:, 0], test_idx[:, 1]] = float('nan')

    train_mask = ~torch.isnan(X_train)
    test_ratings = X[test_idx[:, 0], test_idx[:, 1]]

    return X_train, train_mask, test_idx, test_ratings
