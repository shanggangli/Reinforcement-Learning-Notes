#-*- codeing=utf-8 -*-
#@time: 2020/10/30 0:58
#@Author: Shang-gang Lee
import gym
from regularActorCriticGym import Actor,Critic
import matplotlib.pyplot as plt
import numpy as np
import torch
np.random.seed(2)
torch.manual_seed(1) # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = True  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_states = env.observation_space.shape[0]
N_actions = env.action_space.n
print(N_states)
print(N_actions)

actor = Actor(n_actions=N_actions,n_states=N_states,LreaningRate=LR_A)
critic = Critic(n_states=N_states, LreaningRate=LR_C,reward_decay=GAMMA)     # we need a good teacher, so the teacher should learn faster than the actor


for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break