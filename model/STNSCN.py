# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : model.py
Project      : AAAI2023-STNSCN
Created Date : 2021/11/14 23:21
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

from model.Decoder import Decoder
from model.Encoder import Encoder
from model.Transform import Transform


class STNSCN(nn.Module):
    def __init__(self, input_dim, time_dim, hidden_dim, output_dim, gcn_depth, fusion_mode,
                 num_of_weeks, num_of_days, num_of_hours, num_for_predict, num_for_target, num_of_head,
                 dropout_prob, dropout_type, device, alpha=1, use_transform=True,
                 static_norm_adjs=None,
                 norm='D-1',
                 use_curriculum_learning=True, cl_decay_steps=4000):

        super(STNSCN, self).__init__()

        self.in_channels = input_dim
        self.time_channels = time_dim
        self.hidden_channels = hidden_dim
        self.output_channels = output_dim

        self.dropout = nn.Dropout(p=dropout_prob)

        self.seq_length = num_for_predict

        self.use_transform = use_transform
        self.device = device

        node_num = static_norm_adjs[0].shape[0]
        # node_num = 54
        self.node_num = node_num
        self.encoder = Encoder(input_dim,
                               time_dim,
                               hidden_dim,
                               gcn_depth,
                               alpha,
                               num_of_weeks,
                               num_of_days,
                               num_of_hours,
                               num_for_predict,
                               dropout_prob,
                               dropout_type,
                               fusion_mode,
                               node_num,
                               static_norm_adjs,
                               norm,
                               device,
                               )

        self.decoder = Decoder(input_dim,
                               time_dim,
                               hidden_dim,
                               output_dim,
                               gcn_depth,
                               alpha,
                               num_of_weeks,
                               num_of_days,
                               num_of_hours,
                               num_for_predict,
                               dropout_prob,
                               dropout_type,
                               'mix',
                               node_num,
                               static_norm_adjs,
                               norm,
                               use_curriculum_learning,
                               cl_decay_steps)


        if use_transform == True:
            self.transform = Transform(time_dim,
                                       hidden_dim,
                                       num_of_weeks,
                                       num_of_days,
                                       num_of_hours,
                                       num_for_predict,
                                       num_for_target,
                                       num_of_head,
                                       dropout_prob)

    def forward(self, x, x_time, target_time, target_cl=None, task_level=2, global_step=None):

        batch_size, node_num, num_for_predict, dim = x.shape
        if len(x_time.shape) < 4 and len(target_time.shape) < 4:
            x_time = x_time.unsqueeze(dim=1).repeat(1, node_num, 1, 1)
            target_time = target_time.unsqueeze(dim=1).repeat(1, node_num, 1, 1)

        # 编码器输出丢弃， [batch, RNN_layer, node_num, num_pred, hidden_channels]
        outputs, encoder_hiddens = self.encoder(x, x_time, self.seq_length)

        # [batch, RNN_layer, node_num, num_pred, hidden_channels]
        if self.use_transform == True :
            # [batch, RNN_layer, node_num, num_pred, hidden_channels]
            encoder_hiddens_last = encoder_hiddens[:, :, :, -1, :]
            encoder_hiddens = self.transform(encoder_hiddens, x_time, target_time)
            encoder_hiddens = encoder_hiddens + encoder_hiddens_last

        elif self.use_transform == False:
            encoder_hiddens = encoder_hiddens[:, :, :, -1, :]

        # 解码器输入应为前一个解码器的输出，训练阶段使用真实label作为解码器输入，初始输入是用离预测最近的前一个时间点
        # batch_size, node_num, num_for_predict, dim
        GO_decoder_input = torch.zeros((batch_size, node_num, 1, self.in_channels), device=self.device)

        # 计划性学习，逐步增加输入序列长度，并且按照一定几率使用label作为解码器输入，否则使用前一个时间步解码器输出作为下一个时间步的输入
        outputs_final= self.decoder(GO_decoder_input, target_time, target_cl, encoder_hiddens,
                                    task_level, global_step)

        del outputs, encoder_hiddens, GO_decoder_input

        return outputs_final