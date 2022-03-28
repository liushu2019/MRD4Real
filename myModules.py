
import sys
import argparse
import numpy as np
import pandas as pd
import os
import shutil
import logging
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import Tensor
from utilities import *
# https://github.com/idocx/BP_MLL_Pytorch/blob/master/bp_mll.py
# loss function for bp-mll @ Zhang, Min-Ling, and Zhi-Hua Zhou. "Multilabel neural networks with applications to functional genomics and text categorization." IEEE transactions on Knowledge and Data Engineering 18.10 (2006): 1338-1351.

class BPMLLLoss(torch.nn.Module):
    def __init__(self, bias=(1, 1)):
        super(BPMLLLoss, self).__init__()
        self.bias = bias
        assert len(self.bias) == 2 and all(map(lambda x: isinstance(x, int) and x > 0, bias)), \
            "bias must be positive integers"

    def forward(self, c: Tensor, y: Tensor) -> Tensor:
        r"""
        compute the loss, which has the form:
        L = \sum_{i=1}^{m} \frac{1}{|Y_i| \cdot |\bar{Y}_i|} \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
        :param c: prediction tensor, size: batch_size * n_labels
        :param y: target tensor, size: batch_size * n_labels
        :return: size: scalar tensor
        """
        y = y.float()
        y_bar = -y + 1
        y_norm = torch.pow(y.sum(dim=(1,)), self.bias[0])
        y_bar_norm = torch.pow(y_bar.sum(dim=(1,)), self.bias[1])
        assert torch.all(y_norm != 0) or torch.all(y_bar_norm != 0), "an instance cannot have none or all the labels"
        return torch.mean(1 / torch.mul(y_norm, y_bar_norm) * self.pairwise_sub_exp(y, y_bar, c))

    def pairwise_sub_exp(self, y: Tensor, y_bar: Tensor, c: Tensor) -> Tensor:
        r"""
        compute \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
        """
        truth_matrix = y.unsqueeze(2).float() @ y_bar.unsqueeze(1).float()
        exp_matrix = torch.exp(c.unsqueeze(1) - c.unsqueeze(2))
        return (torch.mul(truth_matrix, exp_matrix)).sum(dim=(1, 2))


class I_BPMLLLoss(torch.nn.Module):
    def __init__(self, bias=(1, 1)):
        super(I_BPMLLLoss, self).__init__()
        self.bias = bias
        assert len(self.bias) == 2 and all(map(lambda x: isinstance(x, int) and x > 0, bias)), \
            "bias must be positive integers"

    def forward(self, c: Tensor, y: Tensor) -> Tensor:
        r"""
        compute the loss, which has the form:
        # L = \sum_{i=1}^{m} \frac{1}{|Y_i| \cdot |\bar{Y}_i|} \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
        :param c: prediction tensor, size: batch_size * n_labels
        :param y: target tensor, size: batch_size * n_labels
        :return: size: scalar tensor
        """
        c1, s1 = split_tensor(c)
        y = y.float()
        y_bar = -y + 1
        y_norm = torch.pow(y.sum(dim=(1,)), self.bias[0])
        y_bar_norm = torch.pow(y_bar.sum(dim=(1,)), self.bias[1])
        assert torch.all(y_norm != 0) or torch.all(y_bar_norm != 0), "an instance cannot have none or all the labels"
        return torch.mean(1 / (2 * torch.mul(y_norm, y_bar_norm) + torch.mul(y_norm, y_norm) + torch.mul(y_bar_norm, y_bar_norm)) * (self.pairwise_sub_exp(y, y_bar, c1) + self.pairwise_sub_exp(y_bar, y, s1) + self.valueThreshod_sub_exp(y, c1, s1) + self.valueThreshod_sub_exp(y_bar, s1, c1)))

    def pairwise_sub_exp(self, y: Tensor, y_bar: Tensor, c: Tensor) -> Tensor:
        r"""
        compute \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
        """
        truth_matrix = y.unsqueeze(2).float() @ y_bar.unsqueeze(1).float()
        exp_matrix = torch.exp(c.unsqueeze(1) - c.unsqueeze(2))
        return (torch.mul(truth_matrix, exp_matrix)).sum(dim=(1, 2))

    def valueThreshod_sub_exp(self, y: Tensor, c: Tensor, s: Tensor) -> Tensor:
        r"""
        compute \sum_{r \in Y_p}\sum_{t \in Y_p}\exp{-(c_{2r}^p-s_{2r+1}^p)}
        """
        truth_matrix = y.unsqueeze(2).float() * y.unsqueeze(1).float()
        exp_matrix = torch.exp(s.unsqueeze(1) - c.unsqueeze(2))
        return (torch.mul(truth_matrix, exp_matrix)).sum(dim=(1, 2))

class RoleModel(nn.Module):
    def __init__(self, init_features, dr_rate, class_num):
        super().__init__()
        NODE_NUM = init_features.shape[0]
        emb_size = init_features.shape[1]

        hidden_sizes = [emb_size, emb_size//2, emb_size//4]

        self.embed = nn.Embedding(NODE_NUM, hidden_sizes[0])
        self.embed.weight.data.copy_(torch.from_numpy(init_features))
        self.linear1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dr_rate = dr_rate
        self.linear2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.linear3 = nn.Linear(hidden_sizes[2], class_num)

    def forward(self, X):
        X = self.embed(X)
        X = F.relu(self.linear1(X))
        X = F.dropout(X, self.dr_rate, training=self.training)
        X = F.relu(self.linear2(X))
        X = F.dropout(X, self.dr_rate, training=self.training)
        # X = F.log_softmax(self.linear3(X),dim=1) # S.Liu
        X = torch.sigmoid(self.linear3(X))
        return X

class RoleModel_ibpmll(nn.Module):
    def __init__(self, init_features, dr_rate, class_num):
        super().__init__()
        NODE_NUM = init_features.shape[0]
        emb_size = init_features.shape[1]

        # hidden_sizes = [emb_size, emb_size//2, emb_size//4]
        hidden_sizes = [emb_size, (emb_size + class_num*2)//2, (emb_size + class_num*8)//5]

        self.embed = nn.Embedding(NODE_NUM, hidden_sizes[0])
        self.embed.weight.data.copy_(torch.from_numpy(init_features))
        self.linear1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dr_rate = dr_rate
        self.linear2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.linear3 = nn.Linear(hidden_sizes[2], class_num*2)

    def forward(self, X):
        X = self.embed(X)
        X = F.relu(self.linear1(X))
        X = F.dropout(X, self.dr_rate, training=self.training)
        X = F.relu(self.linear2(X))
        X = F.dropout(X, self.dr_rate, training=self.training)
        # X = F.log_softmax(self.linear3(X),dim=1) # S.Liu
        X = torch.sigmoid(self.linear3(X))
        return X


class Discriminator(nn.Module):
    TRAIN_DISCRIMINATOR = 'd2'
    def __init__(self, dr_rate, emb_size, discriminatorTpye):
        super().__init__()
        self.TRAIN_DISCRIMINATOR = discriminatorTpye
        hidden_sizes = [emb_size, emb_size//2, emb_size//4]
        self.linear1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dr_rate = dr_rate
        self.linear2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        if self.TRAIN_DISCRIMINATOR == 'd3':
            self.linear3 = nn.Linear(hidden_sizes[2], 3)
        else:
            self.linear3 = nn.Linear(hidden_sizes[2], 1)

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.dropout(X, self.dr_rate, training=self.training)
        X = F.relu(self.linear2(X))
        # X = F.sigmoid(self.linear3(X))
        if self.TRAIN_DISCRIMINATOR == 'd3':
            X = F.log_softmax(self.linear3(X),dim=1)
        else:
            X = torch.sigmoid(self.linear3(X)) # S.Liu
        return X


class SingleRoleModel(nn.Module):
    def __init__(self, dr_rate, emb_size, class_num):
        super().__init__()
        hidden_sizes = [emb_size, emb_size//2, emb_size//4]

        self.linear1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dr_rate = dr_rate
        self.linear2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.linear3 = nn.Linear(hidden_sizes[2], class_num)

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.dropout(X, self.dr_rate, training=self.training)
        X = F.relu(self.linear2(X))
        X = F.dropout(X, self.dr_rate, training=self.training)
        # X = F.log_softmax(self.linear3(X))
        X = torch.sigmoid(self.linear3(X))
        return X

