# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : tester.py
Project      : AAAI2023-STNSCN
Created Date : 2021/11/14 22:14
Author       : Yu Zhao 
Email        : yzhao@buaa.edu.cn
==================================================================
Descriptions :

==================================================================
TODO List:
   Date      	           Comments                        Finish
   
   
   
---------   --------------------------------------------   -------   
"""

import copy
import torch
import numpy as np
import argparse
import time

import os

from tqdm import tqdm

from config.config import *
from model.helper import Trainer
from tools.metrics import metric


def model_val(runid, engine, dataloader, device, logger, epoch):
    logger.info('Start validation phase.....')

    val_dataloder = dataloader['val']

    valid_loss_list = []
    valid_mae_list = []
    valid_mape_list = []
    valid_rmse_list = []
    valid_outputs_list = []

    val_tqdm_loader = tqdm(enumerate(val_dataloder))
    for iter, (x, x_time, target, target_time, pos, target_cl) in val_tqdm_loader:

        x = x.to(engine.device)
        x_time = x_time.to(engine.device)
        target = target.to(engine.device)
        target_time = target_time.to(engine.device)

        loss, mae, mape, rmse, predict = engine.eval(input=x,
                                                     input_time=x_time,
                                                     target=target,
                                                     target_time=target_time)

        valid_loss_list.append(loss)
        valid_mae_list.append(mae)
        valid_mape_list.append(mape)
        valid_rmse_list.append(rmse)
        valid_outputs_list.append(predict)


    mval_loss = np.mean(valid_loss_list)
    mval_mae = np.mean(valid_mae_list)
    mval_mape = np.mean(valid_mape_list)
    mval_rmse = np.mean(valid_rmse_list)
    predicts = torch.cat(valid_outputs_list, dim=0)
    log = 'Evaluate model on validation data, Loss:{:.4f}, Val MAE:{:.4f}, Val MAPE:{:.4f}, Val RMSE:{:.4f}'
    logger.info(log.format(mval_loss, mval_mae, mval_mape, mval_rmse))


    return mval_loss, mval_mae, mval_mape, mval_rmse, predicts


def model_test(runid, engine, dataloader, device, logger, cfg, mode='Test'):
    logger.info('Start testing phase.....')

    test_dataloder = dataloader['test']
    engine.model.eval()

    test_loss_list = []
    test_mae_list = []
    test_mape_list = []
    test_rmse_list = []
    test_outputs_list = []
    test_target_list = []

    test_tqdm_loader = tqdm(enumerate(test_dataloder))
    for iter, (x, x_time, target, target_time, pos, target_cl) in test_tqdm_loader:

        x = x.to(engine.device)
        x_time = x_time.to(engine.device)
        target = target.to(engine.device)
        target_time = target_time.to(engine.device)
        loss, mae, mape, rmse, predict = engine.eval(input=x,
                                                     input_time=x_time,
                                                     target=target,
                                                     target_time=target_time)

        test_loss_list.append(loss)
        test_mae_list.append(mae)
        test_mape_list.append(mape)
        test_rmse_list.append(rmse)
        test_outputs_list.append(predict)
        test_target_list.append(target)


    mtest_loss = np.mean(test_loss_list)
    mtest_mae = np.mean(test_mae_list)
    mtest_mape = np.mean(test_mape_list)
    mtest_rmse = np.mean(test_rmse_list)

    predicts = torch.cat(test_outputs_list, dim=0)
    targets = torch.cat(test_target_list, dim=0)
    log = 'Evaluate best model on test data, Test Loss:{:.4f}, Test MAE:{:.4f}, Test MAPE:{:.4f}, Test RMSE:{:.4f}'
    logger.info(log.format(mtest_loss, mtest_mae, mtest_mape, mtest_rmse))

    if mode=='Test':
        # 测试结果保留，用于生成图像
        pred_all = predicts.cpu()
        path_save_pred = os.path.join(cfg['save'], cfg['model_name'], cfg['data']['freq'], 'result_pred')
        if not os.path.exists(path_save_pred):
            os.makedirs(path_save_pred, exist_ok=True)

        name = 'exp{:d}_Test_mae:{:.4f}_mape:{:.4f}_rmse:{:.4f}'. \
            format(cfg['expid'], mtest_mae, mtest_mape, mtest_rmse)
        path = os.path.join(path_save_pred, name)
        np.save(path, pred_all)
        logger.info('result of prediction has been saved, path: {}'.format(path))
        logger.info('shape: ' + str(pred_all.shape))


        # 按照不同时间段测试
        horizon_mae_list = []
        horizon_mape_list = []
        horizon_rmse_list = []

        points_per_hour = cfg['data']['points_per_hour']
        unit = 60//points_per_hour

        tmae, tmape, trmse = metric(predicts, targets)

        for i in range(0, cfg['data']['num_for_target']):

            # 对不同的时间分别计算loss
            pred = predicts[:, :, i, :]
            label = targets[:, :, i, :]
            metrics = metric(pred, label)

            log = 'Evaluate best model on test data for horizon {:d}min, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            logger.info(log.format((i + 1)*unit, metrics[0], metrics[1], metrics[2]))
            horizon_mae_list.append(metrics[0])
            horizon_mape_list.append(metrics[1])
            horizon_rmse_list.append(metrics[2])

    return mtest_loss, mtest_mae, mtest_mape, mtest_rmse, predicts


def baseline_test(runid, model, dataloader, device, logger, cfg):
    scaler = dataloader['scaler']

    engine = Trainer(
        model,
        base_lr=cfg['train']['base_lr'],
        weight_decay=cfg['train']['weight_decay'],
        milestones=cfg['train']['milestones'],
        lr_decay_ratio=cfg['train']['lr_decay_ratio'],
        min_learning_rate=cfg['train']['min_learning_rate'],
        max_grad_norm=cfg['train']['max_grad_norm'],
        cl_decay_steps=cfg['train']['cl_decay_steps'],
        num_for_target=cfg['data']['num_for_target'],
        num_for_predict=cfg['data']['num_for_predict'],
        loss_type=cfg['model']['loss_type'],
        scaler=scaler,
        device=device,
        curriculum_learning=cfg['train']['use_curriculum_learning'],
        new_training=cfg['train']['new_training'],
    )

    best_mode_path = cfg['train']['best_mode']
    logger.info("loading {}".format(best_mode_path))

    save_dict = torch.load(best_mode_path)
    engine.model.load_state_dict(save_dict['model_state_dict'], strict=False)
    logger.info('model load success! {}'.format(best_mode_path))

    # 计算参数数量
    total_param = 0
    logger.info('Net\'s state_dict:')
    for param_tensor in engine.model.state_dict():
        logger.info(param_tensor + '\t' + str(engine.model.state_dict()[param_tensor].size()))
        total_param += np.prod(engine.model.state_dict()[param_tensor].size())
    logger.info('Net\'s total params:{:d}'.format(int(total_param)))

    logger.info('Optimizer\'s state_dict:')
    for var_name in engine.optimizer.state_dict():
        logger.info(var_name + '\t' + str(engine.optimizer.state_dict()[var_name]))

    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info('Number of model parameters is {:d}'.format(int(nParams)))

    mtest_loss, mtest_mae, mtest_mape, mtest_rmse, predicts = model_test(runid, engine, dataloader, device, logger,
                                                                         cfg, mode='Test')
    return mtest_mae, mtest_mape, mtest_rmse, mtest_mae, mtest_mape, mtest_rmse