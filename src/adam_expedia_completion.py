# src/expedia_nn_completion.py

"""
Neural MNL model and matrix completion components for Expedia hotel recommendation.
Includes:
    - GPU device setup
    - Sales class for offer set structure and scoring
    - NeuralNet3Layer: 3-layer feedforward model
    - ExpediaMatrixCompletion: Matrix completion with Adam
    - Utility functions: seed setting, training, evaluation, matrix filling

Designed for reuse in STAT 4830 Expedia group project.
"""

# === Core Libraries ===
import numpy as np
import pandas as pd
import math

# === PyTorch ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

# === Visualization ===
import matplotlib.pyplot as plt
import seaborn as sns

# === Scikit-learn ===
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === Device Configuration ===
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU available:", device)
    torch.cuda.init()
else:
    device = torch.device("cpu")
    print("GPU unavailable: CPU")

# === Sales Class for Utility Modeling ===
class Sales:
    def __init__(self, all_offer_sets, sell_num, mask):
        self.offer_set_list = []
        self.fw = None
        self.scores = None
        self.N_sales = torch.tensor(np.concatenate(sell_num, axis=1), dtype=torch.float64, device=device)
        self.offer_feature = torch.tensor(all_offer_sets, dtype=torch.float64)
        self.original_shape = self.offer_feature.shape
        self.feature_concat = self.offer_feature.reshape(-1, self.offer_feature.shape[2]).to(device)
        self.N = len(all_offer_sets)
        self.mask = torch.tensor(mask, dtype=torch.float64, device=device)
        self.mask_flat = self.mask.reshape((-1,))
        self.masked_feature_concat = self.feature_concat[self.mask_flat == 1]

    def calculate_all_choice_prob(self, W):
        rule_feature = self.feature_concat
        self.fw = self.calculate_choice_prob(rule_feature, W)
        return self.fw

    def calculate_choice_prob(self, rule_feature, W):
        Z = W(rule_feature).reshape(self.original_shape[:2])
        masked_Z = Z.masked_fill(self.mask == 0, float('-inf'))
        softmax_result = F.softmax(masked_Z, dim=-1).reshape((-1, 1))[self.mask_flat == 1]
        return softmax_result

    def calculate_all_choice_score(self, W):
        rule_feature = self.feature_concat
        self.scores = self.calculate_choice_score(rule_feature, W)
        return self.scores

    def calculate_choice_score(self, rule_feature, W):
        Z = W(rule_feature).reshape(self.original_shape[:2])
        masked_Z = Z.masked_fill(self.mask == 0, float('-inf'))
        masked_result = masked_Z.reshape((-1, 1))[self.mask_flat == 1]
        return masked_result

# === Neural Net Model: 3-Layer Feedforward ===
class NeuralNet3Layer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# === Expedia Matrix Completion with Adam ===
class ExpediaMatrixCompletion(nn.Module):
    def __init__(self, X, train_mask, k):
        super().__init__()
        self.device = X.device
        self.X = X
        self.mask = train_mask
        self.k = k
        self.m, self.n = X.shape
        self.n_obs = train_mask.sum().item()

        # latent factors
        self.U = nn.Parameter(torch.randn(self.m, k, device=self.device) * 0.01)
        self.V = nn.Parameter(torch.randn(self.n, k, device=self.device) * 0.01)

        # bias terms
        mean_value = torch.nanmean(X).item()
        self.b = nn.Parameter(torch.tensor(mean_value, dtype=torch.float32, device=self.device))
        user_means = torch.nanmean(X, dim=1)
        item_means = torch.nanmean(X, dim=0)
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
            raise ValueError("Unsupported loss type")

        reg = l2_lambda * (self.U.norm() ** 2 + self.V.norm() ** 2) / self.n_obs
        return loss + reg

    def filled_matrix(self, pred_matrix=None):
        if pred_matrix is None:
            pred_matrix = self.forward().detach()
        filled = self.X.clone().detach()
        filled[~self.mask] = pred_matrix[~self.mask]
        return filled

    def evaluate_rmse(self):
        pred = self.forward()[self.mask]
        true = self.X[self.mask]
        return F.mse_loss(pred, true).sqrt().item()

# === Utility: Set Seed ===
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === Create Train-Test Split ===
def create_train_test_split(X, num_test=2500, seed=123):
    observed_idx = torch.nonzero(~torch.isnan(X), as_tuple=False)
    perm = torch.randperm(observed_idx.size(0), generator=torch.Generator().manual_seed(seed))
    test_idx = observed_idx[perm[:num_test]]

    X_train = X.clone()
    X_train[test_idx[:, 0], test_idx[:, 1]] = float('nan')

    train_mask = ~torch.isnan(X_train)
    test_ratings = X[test_idx[:, 0], test_idx[:, 1]]

    return X_train, train_mask, test_idx, test_ratings

# === Train Completion Model ===
def train_completion_model(model, epochs=300, lr=0.01, l2=1e-4, loss_type="mse", verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.train_loss(loss_type=loss_type, l2_lambda=l2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train loss = {loss.item():.4f}, RMSE = {model.evaluate_rmse():.4f}")
    return losses
