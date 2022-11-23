# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : config.py
Project      : AAAI2023-STNSCN
Created Date : 2021/8/10 21:17
Author       : Yu Zhao 
Email        : yzhao@buaa.edu.cn
==================================================================
Descriptions :

==================================================================
TODO List:
   Date      	           Comments                        Finish
   
   
   
---------   --------------------------------------------   -------   
"""
import random
import time
import yaml
import logging
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def read_cfg_file(filename):
    '''
    读取配置文件
    :param filename:
    :return:
    '''
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
    return cfg


def get_logger(log_dir,
               name,
               log_filename='info.log',
               level=logging.INFO,
               write_to_file=True):
    '''
    设置Log格式
    :param log_dir:
    :param name:
    :param log_filename:
    :param level:
    :param write_to_file:
    :return:
    '''
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    if write_to_file is True:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


