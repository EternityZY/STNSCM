# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : helper.py
Project      : AAAI2023-STNSCN
Created Date : 2021/11/14 22:13
Author       : Yu Zhao 
Email        : yzhao@buaa.edu.cn
==================================================================
Descriptions :

==================================================================
TODO List:
   Date      	           Comments                        Finish
   
   
   
---------   --------------------------------------------   -------   
"""

import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import math

from torch.optim.lr_scheduler import MultiStepLR

from tools.metrics import masked_mae_torch, masked_mape_torch, masked_rmse_torch


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


class Trainer():
    def __init__(self,
                 model,
                 base_lr,
                 weight_decay,
                 milestones,
                 lr_decay_ratio,
                 min_learning_rate,
                 max_grad_norm,
                 cl_decay_steps,
                 num_for_target,
                 loss_type,
                 num_for_predict,
                 scaler,
                 device,
                 curriculum_learning=True,
                 new_training=False):

        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.device= device
        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)
        self.scheduler = StepLR2(optimizer=self.optimizer,
                                 milestones=milestones,
                                 gamma=lr_decay_ratio,
                                 min_lr=min_learning_rate)

        # Huber Loss
        if loss_type == 'L1':
            self.loss = nn.L1Loss(reduction='mean').to(device)
        elif loss_type == 'L2':
            self.loss = nn.MSELoss(reduction='mean').to(device)
        elif loss_type == 'Smooth':
            self.loss = nn.SmoothL1Loss(reduction='mean').to(device)

        self.max_grad_norm = max_grad_norm
        self.cl_decay_steps = cl_decay_steps

        self.iter = 0
        self.global_step = 0
        self.task_level = 1
        self.num_for_target = num_for_target
        self.num_for_predict = num_for_predict
        self.curriculum_learning = curriculum_learning
        self.new_training = new_training

    def train(self, input, input_time, target, target_time, target_cl):

        # 迭代次数
        self.iter += 1
        self.global_step += 1

        if self.iter % self.cl_decay_steps == 0:
            if self.new_training:
                self.iter = 0

            if self.task_level < self.num_for_target:
                self.task_level += 1
            else:
                self.iter = self.global_step


        # 一步前向传播
        self.model.train()
        self.optimizer.zero_grad()

        if self.curriculum_learning:
            output = self.model(input,
                                input_time,
                                target_time,
                                target_cl,
                                task_level=self.task_level,
                                global_step=self.iter
                                )

            # [batch, node_num, time_len, 2]
            predict = self.scaler.inverse_transform(output)

            # Huber Loss
            loss = self.loss(predict[:, :, :self.task_level, :],
                             target[:, :, :self.task_level, :])

            mae = masked_mae_torch(predict[:, :, :self.task_level, :],
                                   target[:, :, :self.task_level, :],
                                   null_val=np.inf).item()

            mape = masked_mape_torch(predict[:, :, :self.task_level, :],
                                     target[:, :, :self.task_level, :],
                                     null_val=np.inf).item()

            rmse = masked_rmse_torch(predict[:, :, :self.task_level, :],
                                     target[:, :, :self.task_level, :],
                                     null_val=np.inf).item()

        else:
            # [batch, node_num, time_len, 2]
            output = self.model(input, input_time, target_time,
                                target_cl=target_cl,
                                task_level=self.num_for_target,
                                global_step=self.iter)

            # [batch, node_num, time_len, 2]
            predict = self.scaler.inverse_transform(output)

            loss = self.loss(predict, target)
            mae = masked_mae_torch(predict, target, null_val=np.inf).item()
            mape = masked_mape_torch(predict, target, null_val=np.inf).item()
            rmse = masked_rmse_torch(predict, target, null_val=np.inf).item()

        loss.backward()

        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return loss.item(), mae, mape, rmse


    def eval(self, input, input_time, target, target_time):
        self.model.eval()
        with torch.no_grad():
            # x, x_time, target_time, target_cl, task_level=2, batches_seen=None
            output = self.model(input, input_time, target_time,
                                target_cl=None,
                                task_level=self.num_for_target,
                                global_step=None)

        # [batch, node_num, time_len, 2]
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, target)

        mae = masked_mae_torch(predict, target, null_val=np.inf).item()
        mape = masked_mape_torch(predict, target, null_val=np.inf).item()
        rmse = masked_rmse_torch(predict, target, null_val=np.inf).item()
        return loss.item(), mae, mape, rmse, predict

