#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File    :   methods_code.py
@Version :   1.0
@Author  :   Forrest Stone
@Contact :   ysbrilliant@163.com
@Time    :   2023/02/05 10:07:48
@Desc    :   The implements of diversity methods in recommednation
'''

# here put the import lib
import numpy as np
import math
import torch


def DPP(self, kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    reference: https://github.com/laming-chen/fast-map-dpp/blob/master/dpp_test.py
    paper: Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity
    """
    # kernel_matrix = kernel_matrix.clone().detach().cpu().numpy()
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


# get the kernel_matrix
def get_DPP_kernel_matrix(scores, item_representations, sigma=0.9):
    """Get the kernel matrix for DPP

    Args:
        scores (list): The score of items for users
        item_representations (array): The representations of items
        sigma (float): hyper-parameter. Defaults to 0.9.

    Returns:
        array: The kernel matrix of DPP
    """
    item_matrix = item_representations
    item_size = item_matrix.shape[0]
    # caculate the similarity matrix
    feature_matrix1 = item_matrix / \
        torch.norm(item_matrix, dim=-1, keepdim=True)
    similarity_matrix = torch.mm(feature_matrix1, feature_matrix1.t())
    sigma = sigma
    alpha = sigma / (2 * (1 - sigma))
    scores = torch.sigmoid(scores)
    scores = torch.exp(alpha * scores)
    kernel_matrix = scores.reshape(
        (item_size, 1)) * similarity_matrix * scores.reshape(
        (1, item_size))
    kernel_matrix = kernel_matrix.clone().detach().cpu().numpy()
    return kernel_matrix


def MMR(self, scores, similarity_matrix, lambdaCons=0.5, topk=20):
    """The method of MMR.

    Args:
        scores (list): The score of items for users
        similarity_matrix (array): The similarity matrix of items
        lambdaCons (float): _description_. Defaults to 0.5.
        topk (int): The topk items want to calculate. Defaults to 20.

    Returns:
        list: Topk items of recommednation list
    """
    scores = dict(enumerate(scores))
    s, r = [], list(scores.keys())
    while len(s) < topk:
        score = 0
        selectOne = None
        for i in r:
            firstPart = scores[i]
            secondPart = 0
            for j in s:
                sim2 = similarity_matrix[i][j]
                if sim2 > secondPart:
                    secondPart = sim2
            equationScore = (lambdaCons * firstPart -
                             (1 - lambdaCons) * secondPart)
            if equationScore > score:
                score = equationScore
                selectOne = i
        if selectOne == None:
            selectOne = i
        r.remove(selectOne)
        s.append(selectOne)
    return s[:topk]
