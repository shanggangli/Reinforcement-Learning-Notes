#-*- codeing=utf-8 -*-
#@time: 2020/10/20 22:20
#@Author: Shang-gang Lee

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

np.random.seed(1)
torch.manual_seed(1)
class net(nn.Module):
    def __init__(self,n_actions,n_states):
        super(net, self).__init__()
        self.n_actions=n_actions
        self.n_states=n_states

        self.f1=nn.Linear(n_states,10,bias=True)
        self.f1.weight.data.normal_(0,0.1)

        self.f2=nn.Linear(10,n_actions,bias=True)
        self.f2.weight.data.normal_(0,0.1)

    def forward(self,x):
        x=F.relu(self.f1(x))
        x=self.f2(x)
        return x

class PolicyGradient(object):
    def __init__(self,n_actions,n_states,learning_rate=0.01,reward_decay=0.95):
        self.n_actions=n_actions
        self.n_states=n_states
        self.lr=learning_rate
        self.gamma=reward_decay
        self.ep_obs,self.ep_as,self.ep_rs=[],[],[]
        self.PolicyGradientNet=net(n_actions,n_states)
        self.optimizer=torch.optim.Adam(self.PolicyGradientNet.parameters(),lr=self.lr)
        self.loss_func=nn.CrossEntropyLoss()

    def choose_actions(self,observation):
        act_values=self.PolicyGradientNet(torch.tensor(observation[np.newaxis, :],dtype=torch.float32))
        prob_weight=F.softmax(act_values,dim=1)
        #print(prob_weight)
        action=np.random.choice(range(prob_weight.shape[1]),p=prob_weight.data.numpy().ravel())
        return action

    def store_transition(self,s,a,r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs=np.zeros_like(self.ep_rs)
        running_add=0
        for t in range(0,len(discounted_ep_rs)):
            running_add=running_add*self.gamma+self.ep_rs[t]
            discounted_ep_rs[t]=running_add

        #normaize
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def learn(self):
        #discount and normalize episode reward
        discounted_ep_rs_norm=self.discount_and_norm_rewards()

        #train on episode
        torch_ep_obs=torch.tensor(np.vstack(self.ep_obs),dtype=torch.float32)
        torch_ep_as=torch.tensor(np.array(self.ep_as),dtype=torch.long)
        torch_ep_vt=torch.tensor(np.array(discounted_ep_rs_norm),dtype=torch.float32)

        all_act=self.PolicyGradientNet(torch_ep_obs)
        prob_act=F.softmax(all_act,dim=1)

        #!!!!!!!
        neg_log_prob=self.loss_func(prob_act,torch_ep_as)
        # print(neg_log_prob)
        loss=torch.mean(neg_log_prob*torch_ep_vt)    # modulate the gradient with advantage (PG magic happens right here.)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_obs,self.ep_as,self.ep_rs=[],[],[]

        return discounted_ep_rs_norm