#-*- codeing=utf-8 -*-
#@time: 2020/10/19 15:04
#@Author: Shang-gang Lee

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
#parameters
Batch_size = 32
Lr = 0.01
Epsilon = 0.9 #greedy policy
Gamma = 0.9   #reward discount
Target_replace_iter = 100 #target update frequency
Memory_capacity = 2000
env = gym.make('Pendulum-v0')
env = env.unwrapped
N_actions = 25
N_states = env.observation_space.shape[0]    # 3 states in this env
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

# DQN construction
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(N_states,50)
        self.fc1.weight.data.normal_(0,0.1)  # Weight initialization (normal distribution with mean 0 and variance 0.1)
        self.values=nn.Linear(50,1)
        self.values.weight.data.normal_(0,0.1)
        self.advantage=nn.Linear(50,N_actions)
        self.advantage.weight.data.normal_(0, 0.1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        values = self.values(x)
        #print(values)
        advantage = self.advantage(x)
        #print(advantage)
        Q=values + advantage -torch.mean(advantage,dim=1,keepdim=True)
        return Q


class DuelingDQN:
    def __init__(self):
        self.eval_net,self.target_net = Net(),Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory=np.zeros((Memory_capacity,N_states*2+2))
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(),lr=Lr)
        self.loss_func=nn.MSELoss()


    def choose_action(self,x):
        x=torch.unsqueeze(torch.FloatTensor(x),0)
        if np.random.uniform()<Epsilon:
            Q = self.eval_net.forward(x)
            action = torch.max(Q,1)[1].data.numpy()
            action = action[0]
        else:
            action=np.random.randint(0,N_actions)
        return action

    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,[a,r],s_))
        index=self.memory_counter%Memory_capacity
        self.memory[index,:] = transition
        self.memory_counter += 1

    def learn(self):
        # update target net
        if self.learn_step_counter%Target_replace_iter==0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) # copy eval_net's weight to target net
        self.learn_step_counter += 1

        # sample some memory data to learn
        sample_index = np.random.choice(Memory_capacity,Batch_size)

        b_memory = self.memory[sample_index,:]

        # state
        b_s=torch.FloatTensor(b_memory[:,:N_states])
        # action
        b_a=torch.LongTensor(b_memory[:,N_states:N_states+1].astype(int))
        # reward
        b_r=torch.FloatTensor(b_memory[:,N_states+1:N_states+2])
        # next state
        b_s_=torch.FloatTensor(b_memory[:,-N_states:])

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + Gamma * q_next.max(1)[0].view(Batch_size, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数


duelingDQN=DuelingDQN()
#-------------------train------------------------
def train(dqn):
    acc_r=[0]
    total_steps=0
    s=env.reset()   #state

    while True:
        env.render()
        a=dqn.choose_action(s)  #choose action given s
        f_action = (a - (N_actions - 1) / 2) / ((N_actions - 1) / 4)  # [-2 ~ 2] float actions
        s_,r,done,info=env.step(np.array([f_action]))  #get reward

        r /= 10  # normalize to a range of (-1, 0)
        acc_r.append(r+acc_r[-1])   # accumulated reward
        dqn.store_transition(s,a,r,s_)

        #begin learning
        if dqn.memory_counter>Memory_capacity:
            dqn.learn()

        if total_steps-Memory_capacity> 15000:
            break

        s=s_
        total_steps += 1
    return acc_r

#train
acc_r=train(dqn=duelingDQN)

plt.figure(1)
plt.plot(np.array(acc_r),c='r',label='dueling')
plt.legend(loc='best')
plt.ylabel('accumulated reward')
plt.xlabel('training steps')
plt.grid()
plt.show()