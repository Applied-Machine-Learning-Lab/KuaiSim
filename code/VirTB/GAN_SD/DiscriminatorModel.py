#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2019/12/2 下午7:00

from utils.utils import *


class DiscriminatorModel(nn.Module):
    def __init__(self, dim_user=118, dim_hidden=256, dim_out=1, activation=nn.LeakyReLU):
        super(DiscriminatorModel, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(dim_user, dim_hidden),
            activation(),
            nn.Linear(dim_hidden, dim_out),
            nn.Sigmoid()
        )
        self.discriminator.apply(init_weight)

    def forward(self, x):
        return self.discriminator(x)
