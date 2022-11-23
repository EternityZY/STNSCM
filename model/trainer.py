# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : trainer.py
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

import copy
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
from tqdm import tqdm

from model.tester import model_val, model_test
from model.helper import Trainer


sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
def baseline_train(runid, model,
                   model_name,
                   dataloader,
                   static_norm_adjs,
                   device,
                   logger,
                   cfg):

    print("start training...", flush=True)
    save_path = os.path.join(cfg['save'], cfg['model_name'], cfg['data']['freq'], 'ckpt')
    scaler = dataloader['scaler']

    # 建立训练引擎
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

    if cfg['train']['load_initial']:
        best_mode_path = cfg['train']['best_mode']
        logger.info("loading {}".format(best_mode_path))
        save_dict = torch.load(best_mode_path)
        engine.model.load_state_dict(save_dict['model_state_dict'])
        logger.info('model load success! {}'.format(best_mode_path))

    else:
        logger.info('Start training from scratch!')
        save_dict = dict()

    # 初始化收集器
    begin_epoch = cfg['train']['epoch_start']
    epochs = cfg['train']['epochs']
    tolerance = cfg['train']['tolerance']

    his_loss = []
    val_time = []
    train_time = []
    best_val_loss = float('inf')
    best_epoch = -1
    stable_count = 0
    global_step = 0
    # 初始训练信息
    logger.info('begin_epoch: {}, total_epochs: {}, patient: {}, best_val_loss: {:.4f}'.
                format(begin_epoch, epochs, tolerance, best_val_loss))

    for epoch in range(begin_epoch, begin_epoch + epochs + 1):

        train_loss = []
        train_mape = []
        train_rmse = []
        train_mae = []
        t1 = time.time()

        train_dataloder = dataloader['train']
        train_tqdm_loader = tqdm(enumerate(train_dataloder))

        for iter, (x, x_time, target, target_time, pos, target_cl) in train_tqdm_loader:

            """
            x: [batch, node, num_for_predict*3, dim]         已经标准化
            x_time: [batch, num_for_predict*3, dim]         已经one hot
            
            target: [batch, node, num_for_target, dim]       未标准化
            target_time: [batch, num_for_target, dim]       已经one hot
            
            pos: [batch, 4(week, day, hour, pred)]          
            target_cl: [batch, num_for_target, dim]         已经标准化，用于解码器输入
            
            """
            global_step += 1

            x = x.to(device)
            x_time = x_time.to(device)
            target = target.to(device)
            target_time = target_time.to(device)
            target_cl = target_cl.to(device)

            metrics = engine.train(input=x,
                                   input_time=x_time,
                                   target=target,
                                   target_time=target_time,
                                   target_cl=target_cl)

            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])

            # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 按照epoch下降学习率
        engine.scheduler.step()

        t2 = time.time()
        train_time.append(t2 - t1)

        # 验证阶段
        s1 = time.time()
        valid_loss, valid_mae, valid_mape, valid_rmse, valid_outputs = model_val(runid,
                                                                                 engine=engine,
                                                                                 dataloader=dataloader,
                                                                                 device=device,
                                                                                 logger=logger,
                                                                                 epoch=epoch)
        s2 = time.time()
        val_time.append(s2 - s1)

        # 处理训练和验证结果
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)


        # 打印输出信息
        if (epoch - 1) % cfg['train']['print_every'] == 0:
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            logger.info(log.format(epoch, (s2 - s1)))

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, ' \
                  'Training Time: {:.4f}/epoch, Learning rate: {}'
            logger.info(log.format(epoch, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse,
                                   (t2 - t1), str(engine.scheduler.get_lr())))
            log = 'Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
            logger.info(log.format(epoch, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse))

        # 将最好的模型保存，同时保存优化器参数
        his_loss.append(mvalid_loss)
        if mvalid_loss < best_val_loss:
            best_val_loss = mvalid_loss
            epoch_best = epoch
            stable_count = 0
            save_dict.update(model_state_dict=copy.deepcopy(engine.model.state_dict()),
                             epoch=epoch_best,
                             best_val_loss=best_val_loss,
                             optimizer_state_dict=copy.deepcopy(engine.optimizer.state_dict()))

            ckpt_name = "exp{:s}_epoch{:d}_Val_mae:{:.2f}_mape:{:.2f}_rmse:{:.2f}.pth". \
                format(model_name, epoch, mvalid_mae, mvalid_mape, mvalid_rmse)
            best_mode_path = os.path.join(save_path, ckpt_name)
            torch.save(save_dict, best_mode_path)
            logger.info(f'Better model at epoch {epoch_best} recorded.')
            logger.info('Best model is : {}'.format(best_mode_path))
            logger.info('\n')
        else:
            stable_count += 1
            if stable_count > tolerance:
                break

    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)

    logger.info("Training finished")
    logger.info("The valid loss on best model is {:.4f}, epoch:{:d}".format(round(his_loss[bestid], 4), epoch_best))

    logger.info('Start the model test phase........')
    logger.info("loading the best model for this training phase {}".format(best_mode_path))
    save_dict = torch.load(best_mode_path)
    engine.model.load_state_dict(save_dict['model_state_dict'])

    valid_loss, valid_mae, valid_mape, valid_rmse, valid_outputs = model_val(runid,
                                                                             engine=engine,
                                                                             dataloader=dataloader,
                                                                             device=device,
                                                                             logger=logger,
                                                                             epoch=epoch)

    mtest_loss, mtest_mae, mtest_mape, mtest_rmse, test_outputs = model_test(runid, engine, dataloader, device,
                                                                             logger, cfg, mode='Test')

    return valid_mae, valid_mape, valid_rmse, mtest_mae, mtest_mape, mtest_rmse