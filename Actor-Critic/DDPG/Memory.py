#-*- codeing=utf-8 -*-
#@time: 2020/10/31 10:08
#@Author: Shang-gang Lee
import torch
import numpy as np
class Memory:
    def __init__(self,capacity,dims):
        self.capacity=capacity
        self.data=np.zeros((capacity,dims))
        self.pointer=0

    def store_transitions(self,s,a,r,s_):
        transition=np.hstack((s,a,[r],s_))
        index=self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :]=transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]
