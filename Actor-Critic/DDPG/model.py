#-*- codeing=utf-8 -*-
#@time: 2020/10/31 9:24
#@Author: Shang-gang Lee
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,n_actions,n_states,action_bound):
        super(Actor, self).__init__()
        self.n_actions=n_actions
        self.n_states=n_states
        self.action_bound=action_bound

        self.f1=nn.Linear(n_states,30,bias=True)
        self.f1.weight.data.normal_(0,0.3)
        self.f2=nn.Linear(30,n_actions,bias=True)
        self.f2.weight.data.normal_(0,0.3)

    def farward(self,state):
        x=F.relu(self.f1(state))
        x=F.tanh(self.f2(x))
        x=torch.mul(x,self.action_bound)
        return x

class Critic(nn.Module):
    def __init__(self,n_actions,n_states):
        super(Critic, self).__init__()
        self.n_actions=n_actions
        self.n_states=n_states

        self.fsc1=nn.Linear(n_states,30,bias=True)
        self.fsc1.weight.data.normal_(0, 0.3)

        self.fac1=nn.Linear(n_actions,30,bias=True)
        self.fsc1.weight.data.normal_(0,0.3)

        self.f2 = nn.Linear(30, 1, bias=True)
        self.f2.weight.data.normal_(0, 0.3)

    def farward(self,state,action):
        """
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s1=F.relu(self.fsc1(state))

        a1=F.relu(self.fac1(action))

        x=torch.cat((s1,a1),dim=1)

        x=self.f2(x)

        return x