#-*- codeing=utf-8 -*-
#@time: 2020/10/24 16:30
#@Author: Shang-gang Lee

import gym
from PolicyGradient02 import PolicyGradient
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped
render=False
DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

RL=PolicyGradient(n_actions=env.action_space.n,n_states=env.observation_space.shape[0])
running_reward = None
for i_episode in range(1000):
    observation=env.reset()

    while True:
        if render: env.render()

        action=RL.choose_actions(observation)

        observation_, reward, done, info = env.step(action)  # reward = -1 in all cases

        RL.store_transition(observation,action,reward)
        #print('reward:',reward)
        if done:
            # calculate running reward
            ep_rs_sum = sum(RL.ep_rs)
            running_reward = ep_rs_sum if running_reward is None else running_reward * 0.99 + ep_rs_sum * 0.01

            if running_reward > DISPLAY_REWARD_THRESHOLD: render = True     # rendering

            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()  # train
            if i_episode == 30:
                plt.plot(vt)  # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()


            break
        observation = observation_