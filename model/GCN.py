# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : GCN.py
Project      : AAAI2023-STNSCN
Created Date : 2021/8/27 16:07
Author       : Yu Zhao 
Email        : yzhao@buaa.edu.cn
==================================================================
Descriptions :

==================================================================
TODO List:
   Date      	           Comments                        Finish
   
   
   
---------   --------------------------------------------   -------   
"""

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class dyn_gconv(nn.Module):
    def __init__(self):
        super(dyn_gconv, self).__init__()

    def forward(self, A, x):
        # RNN中的图卷积代替线性变换
        x = torch.einsum('bhw,bwc->bhc', (A, x))
        return x.contiguous()


class static_gconv(nn.Module):
    def __init__(self):
        super(static_gconv, self).__init__()

    def forward(self, A, x):
        x = torch.einsum('hw,bwc->bhc', (A, x))
        return x.contiguous()


class gconv(nn.Module):

    def __init__(self):
        super(gconv, self).__init__()

    def forward(self, A, x):
        x = torch.einsum('hw, bwtc->bhtc', (A, x))
        return x.contiguous()


class linear(nn.Module):
    '''使用1Dconv代替线性变换'''

    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = nn.Linear(c_in, c_out)

    def forward(self, x):

        return F.relu(self.mlp(x), inplace=True)

class GCN(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout_prob, graph_num, type=None):
        super(GCN, self).__init__()
        if type == 'RNN':
            self.dyn_gconv = dyn_gconv()
            self.static_gconv = static_gconv()
            self.mlp = linear((gdep + 1) * c_in, c_out)

            # 根据图的数量建立权重参数 graph_num+1 其中1 表示对原始输入量也保留一个权重参数
            # 1表示输入也需要一个平衡参数， 1表示动态图
            self.weight = nn.Parameter(torch.FloatTensor(graph_num+1+1), requires_grad=True)
            self.weight.data.fill_(1.0)

        elif type == 'common':
            self.gconv = gconv()
            self.mlp = linear((gdep + 1) * c_in, c_out)

            # 1表示输入也需要一个平衡参数，
            self.weight = nn.Parameter(torch.FloatTensor(graph_num+1), requires_grad=True)
            self.weight.data.fill_(1.0)


        self.dropout = nn.Dropout(dropout_prob)
        self.graph_num = graph_num
        self.gdep = gdep
        self.type = type

    def forward(self, x, norm_adj, dyn_norm_adj=None):
        h = x
        out = [h]

        weight = F.softmax(self.weight, dim=0)     # 概率归一化
        # weight = self.weight    # 概率归一化

        if self.type == 'RNN':
            for _ in range(self.gdep):
                h_next = weight[0] * x
                for i in range(0, len(norm_adj)):
                    h_next += weight[i+1] * self.static_gconv(norm_adj[i], h)
                if dyn_norm_adj is not None:
                    h_next += weight[-1] * self.dyn_gconv(dyn_norm_adj, h)

                h = h_next
                out.append(h)

        elif self.type == 'common':
            for _ in range(self.gdep):  # 扩散阶
                h = self.weight[0] * x
                for i in range(1, len(norm_adj)):
                    h += self.weight[i] * self.gconv(norm_adj[i], h)
                out.append(h)

        ho = torch.cat(out, dim=-1)

        ho = self.mlp(ho)  # 全连接处理结果，使用sigmoid的激活函数


        return ho

