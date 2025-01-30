import torch.nn as nn
import torch.nn.functional as F
import torch

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