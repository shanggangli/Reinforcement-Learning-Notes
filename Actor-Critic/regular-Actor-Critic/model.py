#-*- codeing=utf-8 -*-
#@time: 2020/10/30 11:47
#@Author: Shang-gang Lee
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self,n_actions,n_states):
        """
        :param n_actions: input action
        :param n_states: input states
        """
        super(Actor, self).__init__()
        self.n_actions=n_actions
        self.n_states=n_states



