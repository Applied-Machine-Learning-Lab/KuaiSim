#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2019/12/2 下午6:42

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

INT = torch.IntTensor
LONG = torch.LongTensor
BYTE = torch.ByteTensor
FLOAT = torch.FloatTensor

device = torch.device('cpu')
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def init_weight(m):
    if type(m) == nn.Linear:
        size = m.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.fill_(0.0)


def to_device(*args):
    return [x.to(device) for x in args]


def log(content):
    print("=" * 100)
    print(time.strftime('%Y-%m-%d %H:%M:%S',
                        time.localtime(time.time())) + ': ' + content)
    print("=" * 100)


def log_func(func_description=""):
    def exec_func(func):
        def wrapper(*args, **kw):
            print("\n" + "#" * 100)
            print('%s 开始 %s %s():' % (time.strftime('%Y-%m-%d %H:%M:%S',
                                                    time.localtime(time.time())), func_description, func.__name__))

            time_start = time.time()
            res = func(*args, **kw)
            time_end = time.time()

            print('% s %s %s() 完成， 耗时: %s 秒' % (time.strftime('%Y-%m-%d %H:%M:%S',
                                                              time.localtime(time.time())), func_description,
                                                func.__name__,
                                                (time_end - time_start)))
            print("#" * 100 + "\n")
            return res

        return wrapper

    return exec_func
