#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2019/12/2 下午6:40
import sys
sys.path.append('/content/gdrive/MyDrive/KRLBenchmark-main/code/VirTB')
from VirTB.utils.utils import *


class GeneratorModel(nn.Module):
    def __init__(self, dim_user=118, dim_seed=128, dim_hidden=128, activation=nn.ReLU):
        super(GeneratorModel, self).__init__()
        self.dim_seed = dim_seed
        self.generator = nn.Sequential(
            nn.Linear(dim_seed, dim_hidden),
            activation(),
            nn.Linear(dim_hidden, dim_user),
        )
        self.generator.apply(init_weight)

    def forward(self, z):
        return self.generator(z)

    # generate user sample from random seed z
    def generate(self, z=None):
        if z is None:
            z = torch.rand((1, self.dim_seed)).to(device)  # generate 1 random seed
        x = self.get_prob_entropy(self.generator(z))[0]  # softmax_feature
        features_num = 30
        features = [None] * features_num
        features[0] = x[:, 0:9]
        features[1] = x[:, 9:10]
        features[2] = x[:, 10:12]
        features[3] = x[:, 12:14]
        features[4] = x[:, 14:15]
        features[5] = x[:, 15:23]
        features[6] = x[:, 23:24]
        features[7] = x[:, 24:33]
        features[8] = x[:, 33:34]
        features[9] = x[:, 34:41]
        features[10] = x[:, 41:42]
        features[11] = x[:, 42:50]
        features[12] = x[:, 50:52]
        features[13] = x[:, 52:59]
        features[14] = x[:, 59:60]
        features[15] = x[:, 60:61]
        features[16] = x[:, 61:77]
        features[17] = x[:, 77:78]
        features[18] = x[:, 78:81]
        features[19] = x[:, 81:82]
        features[20] = x[:, 82:83]
        features[21] = x[:, 83:90]
        features[22] = x[:, 90:95]
        features[23] = x[:, 95:100]
        features[24] = x[:, 100:103]
        features[25] = x[:, 103:106]
        features[26] = x[:, 106:109]
        features[27] = x[:, 109:112]
        features[28] = x[:, 112:115]
        features[29] = x[:, 115:118]
        one_hot = FLOAT([]).to(device)
        for i in range(features_num):
            tmp = torch.zeros_like(features[i], device=device)
            one_hot = torch.cat((one_hot, tmp.scatter_(1, torch.multinomial(features[i], 1), 1)),
                                dim=-1)  # 根据softmax feature 生成one-hot的feature
        return one_hot, features

    def get_prob_entropy(self, x):
        features_num = 30
        features = [None] * features_num
        features[0] = x[:, 0:9]
        features[1] = x[:, 9:10]
        features[2] = x[:, 10:12]
        features[3] = x[:, 12:14]
        features[4] = x[:, 14:15]
        features[5] = x[:, 15:23]
        features[6] = x[:, 23:24]
        features[7] = x[:, 24:33]
        features[8] = x[:, 33:34]
        features[9] = x[:, 34:41]
        features[10] = x[:, 41:42]
        features[11] = x[:, 42:50]
        features[12] = x[:, 50:52]
        features[13] = x[:, 52:59]
        features[14] = x[:, 59:60]
        features[15] = x[:, 60:61]
        features[16] = x[:, 61:77]
        features[17] = x[:, 77:78]
        features[18] = x[:, 78:81]
        features[19] = x[:, 81:82]
        features[20] = x[:, 82:83]
        features[21] = x[:, 83:90]
        features[22] = x[:, 90:95]
        features[23] = x[:, 95:100]
        features[24] = x[:, 100:103]
        features[25] = x[:, 103:106]
        features[26] = x[:, 106:109]
        features[27] = x[:, 109:112]
        features[28] = x[:, 112:115]
        features[29] = x[:, 115:118]
        entropy = 0.0
        softmax_feature = FLOAT([]).to(device)
        for i in range(features_num):
            softmax_feature = torch.cat([softmax_feature, F.softmax(features[i], dim=1)], dim=-1)
            entropy += -(F.log_softmax(features[i], dim=1) * F.softmax(features[i], dim=1)).sum(dim=1).mean()
        return softmax_feature, entropy

    def load(self, path=None):
        if path is None:
            path = r'./VirTB/model/user_G.pt'
        self.load_state_dict(torch.load(path))
