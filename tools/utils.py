# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : utils.py
Project      : AAAI2023-STNSCN
Created Date : 2021/8/10 11:07
Author       : Yu Zhao 
Email        : yzhao@buaa.edu.cn
==================================================================
Descriptions :

==================================================================
TODO List:
   Date      	           Comments                        Finish
   
   
   
---------   --------------------------------------------   -------   
"""
import time
import math
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sparse
from torch.optim.lr_scheduler import MultiStepLR


class StepLR2(MultiStepLR):
    """StepLR with min_lr"""

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 min_lr=2.0e-6):
        """

        :optimizer: TODO
        :milestones: TODO
        :gamma: TODO
        :last_epoch: TODO
        :min_lr: TODO

        """
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        super(StepLR2, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()
        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])

        else:
            lr_candidate = max(self.min_lr, lr_candidate)

        return lr_candidate


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMaxNormalization:
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''
    def __init__(self, max, min):
        self.max = max
        self.min = min

    def transform(self, data):
        data = 1. * (data - self.min) / (self.max - self.min)
        data = 2. * data - 1.
        return data

    def inverse_transform(self, data):
        data = (data+1)/2
        data = data*(self.max - self.min) + self.min
        return data


class StandardScaler_Torch:
    """
    Standard the input
    """

    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sparse.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def getDistance(point1, point2):

    lat1, lng1, lat2, lng2 = point1[0], point1[1], point2[0], point2[1]
    def rad(d):
        return d * math.pi / 180.0
    EARTH_REDIUS = 6378.137
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(pow(math.sin(a / 2), 2) +
                                math.cos(radLat1) *
                                math.cos(radLat2) *
                                pow(math.sin(b / 2), 2)))
    s = s * EARTH_REDIUS
    return s


# DTW 算法...
def DTW(M1, M2):
    # 初始化数组 大小为 M1 * M2
    M1_len = len(M1)
    M2_len = len(M2)
    cost = [[0 for i in range(M2_len)] for i in range(M1_len)]

    # 两个维数相等的向量之间的距离
    def __distance(w1, w2):
        d = abs(w2 - w1)
        return d

    # 初始化 dis 数组
    dis = []
    for i in range(M1_len):
        dis_row = []
        for j in range(M2_len):
            dis_row.append(__distance(M1[i], M2[j]))
        dis.append(dis_row)

    # 初始化 cost 的第 0 行和第 0 列
    cost[0][0] = dis[0][0]
    for i in range(1, M1_len):
        cost[i][0] = cost[i - 1][0] + dis[i][0]
    for j in range(1, M2_len):
        cost[0][j] = cost[0][j - 1] + dis[0][j]

    # 开始动态规划
    for i in range(1, M1_len):
        for j in range(1, M2_len):
            cost[i][j] = min(cost[i - 1][j] + dis[i][j],
                             cost[i - 1][j - 1] + dis[i][j],
                             cost[i][j - 1] + dis[i][j])
    return cost[M1_len - 1][M2_len - 1]


if __name__ == '__main__':

    x1 = np.random.randn(10, 12)
    x2 = np.random.randn(10, 12)
    l1 = np.random.randn(10, 2)
    l2 = np.random.randn(10, 2)

    dt_list = []
    tt_list = []
    for i in range(10):
        for j in range(10):

            dt1 = time.time()
            getDistance(l1[i], l2[j])
            dt2 = time.time()

            dt_list.append(dt2-dt1)

            tt1 = time.time()
            for _ in range(4):
                DTW(x1[i], x2[j])
            tt2 = time.time()

            tt_list.append(tt2-tt1)

    print('dt:', np.mean(dt_list))
    print('tt: ', np.mean(tt_list))

    print('time: ', ((np.mean(tt_list)+np.mean(dt_list))*6300**2)/3600)
