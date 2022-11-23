# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : Transformer.py
Project      : AAAI2023-STNSCN
Created Date : 2021/8/27 15:59
Author       : Yu Zhao 
Email        : yzhao@buaa.edu.cn
==================================================================
Descriptions :

==================================================================
TODO List:
   Date      	           Comments                        Finish
   
   
   
---------   --------------------------------------------   -------   
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transform(nn.Module):
    def __init__(self, time_channels, hidden_channels,
                 num_of_weeks, num_of_days, num_of_hours, num_for_predict, num_for_target,
                 num_head, dropout_prob):

        super(Transform, self).__init__()

        self.dropout = nn.Dropout(p=dropout_prob)
        self.w_length = num_of_weeks * num_for_predict
        self.d_length = num_of_days * num_for_predict
        self.h_length = num_of_hours * num_for_predict

        self.num_head = num_head
        self.d = hidden_channels // num_head
        self.FC_q = nn.Linear(time_channels, hidden_channels)

        # 按照周期性分开
        self.FC_k = nn.Linear(time_channels*(num_of_weeks+num_of_days+num_of_hours), hidden_channels)
        self.FC_v = nn.Linear(hidden_channels, hidden_channels)

        self.FC = nn.Linear(num_for_target * hidden_channels, hidden_channels)


    def forward(self, encoder_hidden, x_time, target_time):
        '''

        :param encoder_hidden:  [batch_size, rnn_layer, node_num, num_for_predict, hidden_channels]
        :param x_time: [batch, node_num, 18, 67]
        :param target_time: [batch, node_num, 2, 67]
        :return:
        '''

        batch_size, rnn_layer, node_num, num_for_predict, hidden_channels = encoder_hidden.shape

        # 根据周期性，把不同周期的特征合并到维度上
        week_time = x_time[:, :, :self.w_length, :]
        day_time = x_time[:, :, self.w_length:self.w_length + self.d_length, :]
        hour_time = x_time[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]
        x_time = torch.cat([week_time, day_time, hour_time], dim=-1)

        query = self.FC_q(target_time)             # 未来嵌入特征用处查询   查找出历史的哪些与未来相似
        key = self.FC_k(x_time)                    # 历史嵌入特征用于被查询，
        value = self.FC_v(encoder_hidden)          # 历史流量用于加权，转换到未来

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)                   # [K * batch_size, num_nodes, num_pred, d]
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0).permute(0, 1, 3, 2)   # [K * batch_size, num_nodes, d, num_his]

        # [K * batch_size, RNN_layer, num_nodes, num_for_predict, d]
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        # 时间注意力，因此  空间和时间维度需要调换
        attention = torch.matmul(query, key)        # [K * batch_size, num_nodes, num_pred, num_for_predict]
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)    # [K * batch_size, num_nodes, num_pred, num_for_predict]

        # [K * batch_size, RNN_layer, num_nodes, num_pred, num_his]
        attention = attention.unsqueeze(dim=1).repeat((1, rnn_layer, 1, 1, 1))

        X = torch.matmul(attention, value)          # [K * batch_size, rnn_layer, num_nodes, num_pred, d]

        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # [batch_size, rnn_layer, num_nodes, num_pred, d * K]

        # [batch_size, rnn_layer, num_nodes, num_pred, hidden] -> [batch_size, rnn_layer, num_nodes, hidden]
        X = self.FC(X.reshape(batch_size, rnn_layer, node_num, -1))

        X = F.relu(X)

        del query, key, value

        return X


