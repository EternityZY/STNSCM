# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : datasets.py
Project      : AAAI2023-STNSCN
Created Date : 2021/8/26 17:49
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
import copy
import sys

# sys.path.append('../..')
from torch.utils.data import Dataset, DataLoader
from tools.utils import StandardScaler



def load_dataset(dataset_dir,
                 train_batch_size,
                 valid_batch_size=None,
                 test_batch_size=None,
                 logger=None):
    cat_data = np.load(dataset_dir, allow_pickle=True)
    all_data = {
        'train': {
            'x': cat_data['train_x'].transpose((0, 2, 1, 3)),               # [batch, node_num, time, dim]
            'target': cat_data['train_target'].transpose((0, 2, 1, 3)),
            'x_time': cat_data['train_x_time'],
            'target_time': cat_data['train_target_time'],
            'pos': cat_data['train_pos'],  # 预测位置的时间戳
        },
        'val': {
            'x': cat_data['val_x'].transpose((0, 2, 1, 3)),
            'target': cat_data['val_target'].transpose((0, 2, 1, 3)),
            'x_time': cat_data['val_x_time'],
            'target_time': cat_data['val_target_time'],
            'pos': cat_data['val_pos'],  # 预测位置的时间戳
        },
        'test': {
            'x': cat_data['test_x'].transpose((0, 2, 1, 3)),
            'target': cat_data['test_target'].transpose((0, 2, 1, 3)),
            'x_time': cat_data['test_x_time'],
            'target_time': cat_data['test_target_time'],
            'pos': cat_data['test_pos'],  # 预测位置的时间戳
        },
        'time_feature_index': cat_data['time_feature_index'].item(),
        'time_weather_data': cat_data['time_weather_data'],
    }

    scaler = StandardScaler(mean=all_data['train']['x'].mean(),  # 0号维度表示速度特征，1号维度表示时间特征
                            std=all_data['train']['x'].std())

    # 对输入x正则化
    for category in ['train', 'val', 'test']:
        all_data[category]['x'] = scaler.transform(all_data[category]['x'])

    # curriculum learning 课程性学习，解码器中有几率输入label，所以需要对train_target也做标准化
    all_data['train']['target_cl'] = copy.deepcopy(all_data['train']['target'])
    all_data['train']['target_cl'] = scaler.transform(all_data['train']['target_cl'])

    train_dataset = traffic_demand_prediction_dataset(all_data['train']['x'],
                                                      all_data['train']['x_time'],
                                                      all_data['train']['target'],
                                                      all_data['train']['target_time'],
                                                      all_data['train']['pos'],
                                                      all_data['train']['target_cl'],
                                                      )

    val_dataset = traffic_demand_prediction_dataset(all_data['val']['x'],
                                                    all_data['val']['x_time'],
                                                    all_data['val']['target'],
                                                    all_data['val']['target_time'],
                                                    all_data['val']['pos'],
                                                    None,
                                                    )

    test_dataset = traffic_demand_prediction_dataset(all_data['test']['x'],
                                                     all_data['test']['x_time'],
                                                     all_data['test']['target'],
                                                     all_data['test']['target_time'],
                                                     all_data['test']['pos'],
                                                     None,
                                                     )

    dataloader = {}
    dataloader['train'] = DataLoader(dataset=train_dataset, shuffle=True, batch_size=train_batch_size)  # num_workers=16
    dataloader['val'] = DataLoader(dataset=val_dataset, shuffle=False, batch_size=valid_batch_size)
    dataloader['test'] = DataLoader(dataset=test_dataset, shuffle=False, batch_size=test_batch_size)

    dataloader['scaler'] = scaler
    dataloader['time_feature_index'] = all_data['time_feature_index']
    dataloader['time_weather_data'] = all_data['time_weather_data']

    logger.info(('train x', all_data['train']['x'].shape))
    logger.info(('train target', all_data['train']['target'].shape))
    logger.info(('train x time', all_data['train']['x_time'].shape))
    logger.info(('train target time', all_data['train']['target_time'].shape))
    logger.info(('train pos', all_data['train']['pos'].shape))

    logger.info('\n')
    logger.info(('val x', all_data['val']['x'].shape))
    logger.info(('val target', all_data['val']['target'].shape))
    logger.info(('val x time', all_data['val']['x_time'].shape))
    logger.info(('val target time', all_data['val']['target_time'].shape))
    logger.info(('val pos', all_data['val']['pos'].shape))

    logger.info('\n')
    logger.info(('test x', all_data['val']['x'].shape))
    logger.info(('test target', all_data['val']['target'].shape))
    logger.info(('test x time', all_data['val']['x_time'].shape))
    logger.info(('test target time', all_data['val']['target_time'].shape))
    logger.info(('test pos', all_data['val']['pos'].shape))

    logger.info('\n')
    logger.info('scaler.mean : {}, scaler.std : {}'.format(scaler.mean,
                                                       scaler.std))

    logger.info('\n')
    logger.info('time feature index : {}'.format(all_data['time_feature_index']))
    logger.info('time weather data : {}'.format(all_data['time_weather_data']))

    return dataloader


class traffic_demand_prediction_dataset(Dataset):
    def __init__(self, x, x_time, target, target_time, pos, target_cl=None):
        # 输入时间特征做one hot
        x_time = self.__generate_one_hot(x_time)
        target_time = self.__generate_one_hot(target_time)

        self.x = torch.tensor(x).to(torch.float32)
        self.x_time = torch.tensor(x_time).to(torch.float32)
        self.target = torch.tensor(target).to(torch.float32)
        self.target_time = torch.tensor(target_time).to(torch.float32)
        self.pos = torch.tensor(pos)

        if target_cl is not None:
            self.target_cl = torch.tensor(target_cl).to(torch.float32)
        else:
            self.target_cl = None

    def __getitem__(self, item):

        if self.target_cl is not None:
            return self.x[item], self.x_time[item], \
                   self.target[item], self.target_time[item], \
                   self.pos[item], self.target_cl[item]
        else:
            return self.x[item], self.x_time[item], \
                   self.target[item], self.target_time[item], \
                   self.pos[item], self.pos[item]

    def __len__(self):
        return self.x.shape[0]

    def __generate_one_hot(self, arr):
        dayofweek_len = int(arr[:, :, 0].max()) + 1
        timeofday_len = int(arr[:, :, 1].max()) + 1

        dayofweek = np.zeros((arr.shape[0], arr.shape[1], dayofweek_len))
        timeofday = np.zeros((arr.shape[0], arr.shape[1], timeofday_len))

        for i in range(arr.shape[0]):
            dayofweek[i] = np.eye(dayofweek_len)[arr[:, :, 0][i].astype(np.int)]

        for i in range(arr.shape[0]):
            timeofday[i] = np.eye(timeofday_len)[arr[:, :, 1][i].astype(np.int)]
        arr = np.concatenate([dayofweek, timeofday, arr[..., 2:]], axis=-1)
        return arr
