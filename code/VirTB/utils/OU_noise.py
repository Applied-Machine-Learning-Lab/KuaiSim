#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午8:29
import copy

import numpy as np
import torch


class OUNoise:
    def __init__(self, size, scale=1.0, mu=0, theta=0.15, sigma=0.2):
        self.size = size
        self.scale = scale
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=x.shape)
        self.state = x + dx
        return torch.FloatTensor(self.state * self.scale)
