#-*- codeing=utf-8 -*-
#@time: 2020/10/17 15:15
#@Author: Shang-gang Lee

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gym

#parameters
Batch_size = 32
Lr = 0.01
Epsilon = 0.9 #greedy policy
Gamma = 0.9   #reward discount
Target_replace_iter = 100 #target update frequency
Memory_capacity = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_actions = env.action_space.n  # two actions in this env
N_states = env.observation_space.shape[0]    # 4 states in this env
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

# DQN construction
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(N_states,50)
        self.fc1.weight.data.normal_(0,0.1)  # Weight initialization (normal distribution with mean 0 and variance 0.1)
        self.out=nn.Linear(50,N_actions)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value

# we need to define two nets in DQN: one for target,the other for eval
class DQN(object):
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
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value,1)[1].data.numpy()
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

        q_eval_max_action_index = self.eval_net(b_s_).max(1)[1].view(Batch_size, 1)
        #print('####:',q_eval2)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next_max = self.target_net(b_s_).gather(1,q_eval_max_action_index).detach()
        #print('q_next', q_next_max)
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + Gamma * q_next_max
        #print('q_target',q_target)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数

dqn=DQN()

#-------------------train------------------------
for i in range(400):
    print('<<<<<Episode:%s'%i)
    s=env.reset()   #state
    episode_reward_sum=0

    while True:
        env.render()
        a=dqn.choose_action(s)  #choose action given s

        s_,r,done,info=env.step(a)  #get reward

        # update reward
        x,x_dot,theta,theta_dot=s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        new_r = r1 + r2

        dqn.store_transition(s,a,new_r,s_)
        episode_reward_sum+=new_r

        s=s_

        #begin learning
        if dqn.memory_counter>Memory_capacity:
            dqn.learn()

        if done:
            print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
            break