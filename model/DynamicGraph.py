# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : DynamicGraph.py
Project      : AAAI2023-STNSCN
Created Date : 2021/8/27 20:36
Author       : Yu Zhao 
Email        : yzhao@buaa.edu.cn
==================================================================
Descriptions :

==================================================================
TODO List:
   Date      	           Comments                        Finish
   
   
   
---------   --------------------------------------------   -------   
"""
import torch.nn.functional as F
import torch
import torch.nn as nn

import math


class DynamicGraphGenerate(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout_prob, node_num, reduction=16, alpha=3, norm='D-1'):
        super(DynamicGraphGenerate, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.start_FC = nn.Linear(in_channels+hidden_channels, hidden_channels)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.norm = norm
        self.fc = nn.Sequential(
                                nn.Linear(node_num, hidden_channels // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_channels // reduction, node_num, bias=False),
                                nn.Sigmoid()
                            )
        self.dropout = nn.Dropout(dropout_prob)

        # 用于控制激活函数饱和率
        self.alpha = alpha


    def forward(self, input, hidden):

        # 输入维度均为(batch, node_num, hidden_dim)
        x = input
        batch_size, node_num, hidden_dim = x.shape

        node_feature = torch.cat([input, hidden], dim=-1)

        node_feature = self.start_FC(node_feature)

        # 残差连接
        # (batch, node_num, hidden_dim)
        residual = node_feature

        # 对维度全局平均池化，留下顶点数
        # (batch, node_num, hidden_dim) -> (batch, node_num, 1) -> (batch, 1, node_num)
        residual = self.avg_pool(residual).permute((0, 2, 1))
        # (batch, 1, node_num) -> (batch, node_num, 1)
        residual = self.fc(residual).permute((0, 2, 1))

        node_feature = torch.mul(residual.expand_as(node_feature), node_feature)

        # 内积求相似度
        similarity = torch.matmul(node_feature, node_feature.transpose(2, 1)) / math.sqrt(hidden_dim)



        if self.norm == 'D-1':
            adj = F.relu(torch.tanh(self.alpha * similarity))
            norm_adj = adj / torch.unsqueeze(adj.sum(dim=-1), dim=-1)           # 归一化
        elif self.norm=='softmax':
            adj = F.softmax(F.relu(similarity), dim=2)
            norm_adj = adj

        return norm_adj, adj