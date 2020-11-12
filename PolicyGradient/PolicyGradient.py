#-*- codeing=utf-8 -*-
#@time: 2020/10/18 15:42
#@Author: Shang-gang Lee

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import cv2

#Hyperparameters
learning_rate=0.01
gamma=0.99
env=gym.make('CartPole-v1')
env=env.unwrapped()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space=env.observation_space.shape[0]
        self.action_space=env.action_space.n

        self.l1=nn.Linear(self.state_space,128,bias=False)
        self.l1.weight.data.normal_(0,0.1)
        self.l2=nn.Linear(128,self.action_space,bias=False)
        self.l2.weight.data.normal_(0,0.1)

        self.gamma=gamma

    def farward(self,x):
        x=F.relu(self.l1(x))
        action_value=self.l2(x)
        return action_value
policy = Policy()
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

def prepro(img_size):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    img_grey=cv2.cvtColor(img_size,cv2.COLOR_RGB2GRAY)
    img_resize=cv2.resize(img_grey,(80,80))
    return img_resize

def select_action(state):
    





