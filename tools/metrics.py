# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : metrics.py
Project      : Similarity.py
Created Date : 2021/8/11 16:36
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
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

def masked_mse_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels, null_val=null_val))


def masked_mae_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # 方式label存在0，使得除法变成inf，因此labels小于10的不计算mape
    preds = preds[labels > 10]
    mask = mask[labels > 10]
    labels = labels[labels > 10]

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100


def metric(pred, real):
    mae = masked_mae_torch(pred, real, np.inf).item()
    mape = masked_mape_torch(pred, real, np.inf).item()
    rmse = masked_rmse_torch(pred, real, np.inf).item()
    return mae, mape, rmse
