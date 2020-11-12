#-*- codeing=utf-8 -*-
#@time: 2020/10/31 9:24
#@Author: Shang-gang Lee

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import model
class trainer:
    def __init__(self,n_actions,n_states,action_bound,):
