
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

def split_tensor(input: Tensor):
# def split_tensor(input):
    input = input.clone()
    size = list(input.shape)
    assert len(size) == 2
    assert size[-1]%2 == 0
    input = input.reshape(size[:-1]+[size[-1]//2,-1])
    return input[:,:,0],input[:,:,1]


def ml_1sub_hamming_loss(c: Tensor, y: Tensor, threshold=0.5) -> Tensor:
    """
    compute the hamming loss (refer to the origin paper) (reversed)
    :param c: size: batch_size * n_labels, output of NN
    :param y: size: batch_size * n_labels, target
    :return: Scalar
    """
    if isinstance(threshold, int):
        assert 0 <= threshold <= 1, "threshold should be between 0 and 1"
    if isinstance(threshold, Tensor):
        assert threshold.shape == c.shape, "threshold should be same shape with c"
    p, q = c.size()
    return 1 - (1.0 / (p * q) * (((c > threshold).int() - y) != 0).float().sum())
    
def ml_1sub_one_errors(c: Tensor, y: Tensor, threshold=0.5) -> Tensor:
    """
    compute the one-error function (reversed)
    """
    p, _ = c.size()
    g = torch.argmax(c - threshold, dim=1)
    return (y[:, g] * (c >= threshold).int()[:, g]).trace() / p

def ml_subset_accuracy(c: Tensor, y: Tensor, threshold=0.5) -> Tensor:
    """
    compute the subset accuracy function
    the most strict evaluation metric
    """
    return (((c > threshold).int() != y).float().sum(dim=1) == 0).sum() / c.shape[0]

def ml_accuracy(c: Tensor, y: Tensor, threshold=0.5) -> Tensor:
    """
    compute the accuracy function
    proportion between the number of correct labels and the total number of active labels
    """
    return (((c > threshold).int() * y).sum(dim=1) / (((c > threshold).int() + y) > 0).float().sum(dim=1)).sum() / c.shape[0]

def ml_precision(c: Tensor, y: Tensor, threshold=0.5) -> Tensor:
    """
    compute the precision function
    proportion between the number of correct labels and the total number of predicted labels
    """
    d = ((c > threshold).int() * y).sum(dim=1) / (c > threshold).int().sum(dim=1)
    d[d!=d]=0
    return d.sum() / c.shape[0]

def ml_recall(c: Tensor, y: Tensor, threshold=0.5) -> Tensor:
    """
    compute the recall function
    proportion between the number of correct labels and the total number of truly relevant labels
    """
    return (((c > threshold).int() * y).sum(dim=1) / y.sum(dim=1)).sum() / c.shape[0]

def ml_F(c: Tensor, y: Tensor, threshold=0.5) -> Tensor:
    """
    compute the F measure function (F1)
    harmonic mean of recall and precision
    """
    p = ml_precision(c, y, threshold)
    r = ml_recall(c, y, threshold)
#     assert p+r != 0
    return 2*p*r/(p+r)

def get_multi_metrix(c: Tensor, y: Tensor, threshold=0.5):
    """
    get metrixs
    1 - hammingloss, 1 - one-errors, subset-accuracy, accuracy, precision, recall, F1
    """
    return ml_1sub_hamming_loss(c,y,threshold),\
     ml_1sub_one_errors(c,y,threshold), \
     ml_subset_accuracy(c,y,threshold), \
     ml_accuracy(c,y,threshold), \
     ml_precision(c,y,threshold), \
     ml_recall(c,y,threshold), \
     ml_F(c,y,threshold)


def exec_tsne(train_X, test_X):
    train_shape = train_X.shape
    X = np.concatenate([train_X, test_X], axis=0)

    assert(X.shape[0] == train_X.shape[0] + test_X.shape[0])

    tsne = TSNE(n_components=2)
    tsne_X = tsne.fit_transform(X)
    tsne_train_X, tsne_test_X = tsne_X[:train_shape[0], :], tsne_X[train_shape[0]:, :]

    assert(tsne_train_X.shape[0] == train_X.shape[0])

    return tsne_train_X, tsne_test_X


def exec_pca(train_X, test_X):
    pca = PCA(n_components=2)
    pca_train_X = pca.fit_transform(train_X)
    pca_test_X = pca.transform(test_X)

    return pca_train_X, pca_test_X

def exec_pca_v(train_X, test_X, val_X):
    pca = PCA(n_components=2)
    pca_train_X = pca.fit_transform(train_X)
    pca_val_X = pca.transform(val_X)
    pca_test_X = pca.transform(test_X)

    return pca_train_X, pca_test_X, pca_val_X

def plot_two_scatter(train_X, test_X):
    plt.figure(figsize=(10, 10))
    plt.scatter(x=train_X[:, 0], y=train_X[:, 1], c="blue")
    plt.scatter(x=test_X[:, 0],  y=test_X[:, 1], c="red")


def get_f1(output: torch.tensor, labels: torch.tensor) -> float:
    """
    INPUT:
        output: (N, CLASS_LABEL_NUM)  # one-hot expression
        labels (N, )  # raw expression
    OUTPUT:
        micro_f1: float
        macro_f1: float
    """
    from sklearn.metrics import f1_score
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    output = np.argmax(output, axis=1)
    assert(output.shape[0] == labels.shape[0])
    micro_f1 = f1_score(labels, output, average="micro")
    macro_f1 = f1_score(labels, output, average="macro")
    return micro_f1, macro_f1
    
def get_accuracy_multilabels(output: torch.tensor, labels: torch.tensor) -> float:
    acc_percent = float(torch.round(output).eq(labels).sum().numpy() / len(labels.flatten()))
    acc_perfect = np.mean([ 1 if torch.round(output[i]).eq(labels[i]).sum().numpy() == labels.shape[1] else 0 for i in range(len(labels)) ])
    return acc_percent, acc_perfect

def get_accuracy_multilabels_ibpmll(output: torch.tensor, threshold: torch.tensor, labels: torch.tensor) -> float:
    acc_percent = float((output > threshold).float().eq(labels).sum().numpy() / len(labels.flatten()))
    acc_perfect = np.mean([ 1 if (output > threshold).float()[i].eq(labels[i]).sum().numpy() == labels.shape[1] else 0 for i in range(len(labels)) ])
    return acc_percent, acc_perfect

def get_accuracy(output: torch.tensor, labels: torch.tensor) -> float:
    """
    INPUT:
        output: (N, CLASS_LABEL_NUM)  # one-hot expression
        labels (N, )  # raw expression
    OUTPUT:
        accuracy: float
    """
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    output = np.argmax(output, axis=1)
    assert(output.shape[0] == labels.shape[0])
    accuracy = np.sum(output == labels) / labels.shape[0]
    return accuracy


def accuracy(output: torch.tensor, labels: torch.tensor):
    """
    INPUT:
        output: (N, 1), labels: (N, 1)
    OUTPUT:
        accuracy: float
    """
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    # TODO: Flattenいる？
    output = output.flatten()
    print (output.shape, labels.shape[0])
    assert(output.shape[0] == labels.shape[0])
    output = output > 0.5
    accuracy = np.sum(output == labels) / labels.shape[0]

    return accuracy


def split_cross_val(X, cv_num=5):
    data_num = X.shape[0]
    devided_id = np.arange(0, data_num)
    np.random.shuffle(devided_id)
    cv_size = data_num // cv_num
    train_val_dicts = {"train": [], "test": []}  # {train: [[] * 5,] test: [[] * 5]}
    for i in range(cv_num):
        val_id = devided_id[i * cv_size: (i + 1) * cv_size] if i != cv_num - 1 else devided_id[i * cv_size:]
        train_id = [id_ for id_ in devided_id if id_ not in val_id]
        train_val_dicts["train"].append(train_id)
        train_val_dicts["test"].append(val_id)
    return train_val_dicts