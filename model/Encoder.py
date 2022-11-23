# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : Encoder.py
Project      : AAAI2023-STNSCN
Created Date : 2021/8/28 13:52
Author       : Yu Zhao 
Email        : yzhao@buaa.edu.cn
==================================================================
Descriptions :

==================================================================
TODO List:
   Date      	           Comments                        Finish
   
   
   
---------   --------------------------------------------   -------   
"""

import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
import math
import time

import sys

from model.GraphGateRNN import GraphGateRNN


class Encoder(nn.Module):

    def __init__(self, in_channels, time_channels, hidden_channels, gcn_depth, alpha,
                 num_of_weeks, num_of_days, num_of_hours, num_for_predict, dropout_prob, dropout_type,
                 fusion_mode,
                 node_num,
                 static_norm_adjs,
                 norm, device):
        super(Encoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout_prob)
        self.w_length = num_of_weeks * num_for_predict
        self.d_length = num_of_days * num_for_predict
        self.h_length = num_of_hours * num_for_predict

        self.seq_length = num_for_predict

        self.static_norm_adjs = static_norm_adjs

        self.in_channels = in_channels
        self.time_channels = time_channels
        self.hidden_channels = hidden_channels
        self.RNN_layer = 1
        self.device = device

        self.RNNCell = nn.ModuleList([
                                         GraphGateRNN(in_channels,
                                                      time_channels,
                                                      hidden_channels,
                                                      dropout_type=dropout_type,
                                                      gcn_depth=gcn_depth,
                                                      alpha=alpha,
                                                      num_of_weeks=num_of_weeks,
                                                      num_of_days=num_of_days,
                                                      num_of_hours=num_of_hours,
                                                      dropout_prob=dropout_prob,
                                                      fusion_mode=fusion_mode,
                                                      node_num=node_num,
                                                      norm=norm,
                                                      static_norm_adjs=static_norm_adjs)
                                     ])

    def forward(self, input, x_time, seq_length):
        """

        :param input: [batch, node, num_pred*3, input_dim]
        :param x_time: [batch, node, num_pred*3, time_dim]
        :param seq_length: num_pred
        :return: [batch, node, num_pred, hidden_dim]
        """
        # 注意周期性分开处理，因此编码器中时间步长度应该为 num_for_predict的长度
        x = input
        batch_size, node_num, time_len, dim = x.shape
        Hidden_State = [self.initHidden(batch_size, node_num, self.hidden_channels) for _ in range(self.RNN_layer)]

        week_feature = x[:, :, :self.w_length, :]
        week_time = x_time[:, :, :self.w_length, :]

        day_feature = x[:, :, self.w_length:self.w_length + self.d_length, :]
        day_time = x_time[:, :, self.w_length:self.w_length + self.d_length, :]

        hour_feature = x[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]
        hour_time = x_time[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]

        outputs = []
        hiddens = []

        for i in range(seq_length):  # GRU   编码器过程
            input_cur = torch.cat([week_feature[:, :, i:i + 1, :],
                                   day_feature[:, :, i:i + 1, :],
                                   hour_feature[:, :, i:i + 1, :]], dim=2)

            input_time = torch.cat([week_time[:, :, i:i + 1, :],
                                    day_time[:, :, i:i + 1, :],
                                    hour_time[:, :, i:i + 1, :]], dim=2)

            for j, rnn_cell in enumerate(self.RNNCell):
                # 编码器输入特征维度为[batch, node_num, 3, in_channels]
                cur_h = Hidden_State[j]
                cur_out, cur_h = rnn_cell(input_cur, input_time, cur_h)

                Hidden_State[j] = cur_h
                input_cur = F.relu(cur_out, inplace=True)
                input_time = None

            outputs.append(cur_out.unsqueeze(dim=2))

            # [batch, node_num, hidden_channels] * RNN_layer -> [batch, RNN_layer, node_num, hidden_channels]
            # ->[batch, RNN_layer, node_num, time_len, hidden_channels]
            hidden = torch.stack(Hidden_State, dim=1).unsqueeze(dim=3)
            hiddens.append(hidden)

        # [batch, num_node, time_len, hidden_channels]
        outputs = torch.cat(outputs, dim=2)
        hiddens = torch.cat(hiddens, dim=3)

        # 编码器输出被抛弃，最后一个隐藏层作为解码器隐藏层的初始输入
        return outputs, hiddens

    def initHidden(self, batch_size, num_nodes, hidden_dim):
        '''
        建立RNN初始出入的隐藏层，该隐藏层不需要计算梯度
        :param batch_size:
        :param num_nodes:
        :param hidden_dim:
        :return:
        '''
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(
                torch.zeros((batch_size, num_nodes, hidden_dim)).to(self.device))
            return Hidden_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_dim))
            return Hidden_State
