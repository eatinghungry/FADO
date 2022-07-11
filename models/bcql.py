from os.path import join
import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from zmq import device
import copy


BATCH_SIZE = 32                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
N_ACTIONS = 8


class BCQL(object):
    def __init__(self, ):
        pass