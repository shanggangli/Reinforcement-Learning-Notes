#-*- codeing=utf-8 -*-
#@time: 2020/10/29 21:25
#@Author: Shang-gang Lee

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actornet(nn.Module):
    def __init__(self,n_actions,n_states):
        super(Actornet, self).__init__()
        self.n_actions=n_actions
        self.n_states=n_states

        self.f1=nn.Linear(n_states,24)
        self.f1.weight.data.normal_(0,0.1)
        self.f2=nn.Linear(24,n_actions)
        self.f2.weight.data.normal_(0,0.1)

    def forward(self,x):
        x=F.relu(self.f1(x))
        x=self.f2(x)
        return x

class CirticNet(nn.Module):
    def __init__(self,n_states):
        super(CirticNet, self).__init__()
        self.n_states=n_states

        self.f3=nn.Linear(n_states,24)
        self.f3.weight.data.normal_(0,0.1)
        self.f4=nn.Linear(24,1) #for critic network
        self.f4.weight.data.normal_(0,0.1)
    def forward(self,x):
        x=F.relu(self.f3(x))
        x=self.f4(x)
        return x


class Actor(object):
    def __init__(self,n_actions,n_states,LreaningRate=0.01):
        self.n_actions=n_actions
        self.n_states=n_states
        self.lr=LreaningRate
        self.ActorNet=Actornet(n_actions,n_states)
        self.optimizer1=torch.optim.Adam(self.ActorNet.parameters(),lr=self.lr)

    def learn(self,s,a,td_error):
        s = s[np.newaxis, :]
        s=torch.tensor(s,dtype=torch.float32)
        acts_value=self.ActorNet(s)

        log_prob=F.log_softmax(acts_value,dim=1)
        log_prob=log_prob[0,a]

        exp_v=torch.mean(-log_prob*td_error.detach())

        self.optimizer1.zero_grad()
        exp_v.backward()
        self.optimizer1.step()
        return exp_v

    def choose_action(self,s):
        s=s[np.newaxis, :]
        s=torch.tensor(s,dtype=torch.float32)
        acts_value = self.ActorNet(s)
        prob = F.softmax(acts_value, dim=1)
        return np.random.choice(np.arange(prob.shape[1]),p=prob.data.numpy().ravel())

class Critic(object):
    def __init__(self,n_states,reward_decay,LreaningRate=0.01):
        self.n_states=n_states
        self.gamma = reward_decay
        self.lr=LreaningRate

        self.CirticNet = CirticNet(self.n_states)
        self.optimizer2 = torch.optim.Adam(self.CirticNet.parameters(), lr=self.lr)

    def learn(self,s,r,s_):
        s, s_=s[np.newaxis, :], s_[np.newaxis, :]
        s,s_=torch.tensor(s,dtype=torch.float32),torch.tensor(s_,dtype=torch.float32)

        v_=self.CirticNet(s_)
        v=self.CirticNet(s)
        td_error=r + self.gamma*v_ - v
        loss=torch.square(td_error)

        self.optimizer2.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer2.step()
        return loss




