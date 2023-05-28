#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2019/12/2 下午7:14
from GAN_SD.DiscriminatorModel import DiscriminatorModel
from GAN_SD.GeneratorModel import GeneratorModel
from utils.utils import *


class GanSDModel:
    def __init__(self, dim_user, dim_seed, lr_g, lr_d, expert_users, batch_size=256, alpha=1.0, beta=1.0):
        self.dim_user = dim_user
        self.dim_seed = dim_seed

        self.expert_users = expert_users[:, :118]
        self.batch_size = batch_size
        self.n_expert_users = self.expert_users.size(0)

        self.alpha = alpha
        self.beta = beta

        self.G = GeneratorModel(dim_user=dim_user,activation=nn.Tanh)
        self.D = DiscriminatorModel(dim_user=dim_user,activation=nn.LeakyReLU)

        self.optim_G = optim.Adam(self.G.parameters(), lr=lr_g)
        self.optim_D = optim.Adam(self.D.parameters(), lr=lr_d)
        self.loss_func = nn.BCELoss()

        to_device(self.G, self.D, self.loss_func)

    @log_func("Train GAN-SD Model")
    def train(self):
        n_batch = (self.n_expert_users + self.batch_size - 1) // self.batch_size

        time_start = time.time()
        writer = SummaryWriter()

        for epoch in range(2):
            idx = torch.randperm(self.n_expert_users)
            for i in range(n_batch):

                # sample minibatch from expert users
                batch_expert = self.expert_users[idx[i * self.batch_size:(i + 1) * self.batch_size]]
                # sample minibatch from generator
                batch_seed = torch.normal(torch.zeros(batch_expert.size(0), self.dim_seed),
                                          torch.ones(batch_expert.size(0), self.dim_seed)).to(device)
                batch_gen, _ = self.generate(batch_seed)

                # gradient ascent update discriminator
                for _ in range(1):
                    self.optim_D.zero_grad()

                    expert_o = self.D(batch_expert.to(device))
                    gen_o = self.D(batch_gen.detach())

                    d_loss = self.loss_func(expert_o, torch.ones_like(expert_o, device=device)) + \
                             self.loss_func(gen_o, torch.zeros_like(gen_o, device=device))
                    d_loss.backward()

                    self.optim_D.step()

                # gradient ascent update generator
                for _ in range(10):
                    self.optim_G.zero_grad()
                    # sample minibatch from generator
                    batch_seed = torch.normal(torch.zeros(batch_expert.size(0), self.dim_seed),
                                              torch.ones(batch_expert.size(0), self.dim_seed)).to(device)
                    batch_gen, batch_gen_feature = self.generate(batch_seed)
                    gen_o = self.D(batch_gen.detach())

                    kl = self.get_kl(batch_gen_feature, batch_expert)
                    # g_loss = -(gen_o.sum() +
                    #          self.alpha * self.get_prob_entropy(batch_gen)[1] -
                    #          self.beta * kl)
                    g_loss = self.loss_func(gen_o, torch.ones_like(gen_o, device=device)) + self.beta * kl - \
                             self.alpha * self.get_prob_entropy(batch_gen)[1]
                    g_loss.backward()
                    self.optim_G.step()

                writer.add_scalars('GAN_SD/train_loss', {'discriminator_GAN_SD': d_loss,
                                                         'generator_GAN_SD': g_loss,
                                                         'KL_divergence': kl},
                                   epoch * n_batch + i)

                if i % 10 == 0:
                    cur_time = time.time() - time_start
                    eta = cur_time / (i + 1) * (n_batch - i - 1)
                    print('Epoch %2d Batch %4d G_Loss %.3f KL %.3f D_Loss %.3f. Time elapsed: %.2fs ETA : %.2fs' % (
                        epoch, i, g_loss.cpu().detach().numpy(), kl.cpu().detach().numpy(),
                        d_loss.cpu().detach().numpy(), cur_time, eta))
            if (epoch + 1) % 50 == 0:
                self.save_model()

    # generate random user with one-hot encoded feature
    def generate(self, z=None):
        if z is None:
            z = torch.rand((1, self.dim_seed)).to(device)  # generate 1 random seed
        x = self.get_prob_entropy(self.G(z))[0]  # softmax_feature
        # features = [None] * 11
        # features[0] = x[:, 0:8]
        # features[1] = x[:, 8:16]
        # features[2] = x[:, 16:27]
        # features[3] = x[:, 27:38]
        # features[4] = x[:, 38:49]
        # features[5] = x[:, 49:60]
        # features[6] = x[:, 60:62]
        # features[7] = x[:, 62:64]
        # features[8] = x[:, 64:67]
        # features[9] = x[:, 67:85]
        # features[10] = x[:, 85:88]
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
        # features[0] = x[:, 0:8]
        # features[1] = x[:, 8:16]
        # features[2] = x[:, 16:27]
        # features[3] = x[:, 27:38]
        # features[4] = x[:, 38:49]
        # features[5] = x[:, 49:60]
        # features[6] = x[:, 60:62]
        # features[7] = x[:, 62:64]
        # features[8] = x[:, 64:67]
        # features[9] = x[:, 67:85]
        # features[10] = x[:, 85:88]
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

    def get_kl(self, batch_gen_feature, batch_expert):
        # distributions = [None] * 11
        # distributions[0] = batch_expert[:, :8]
        # distributions[1] = batch_expert[:, 8:16]
        # distributions[2] = batch_expert[:, 16:27]
        # distributions[3] = batch_expert[:, 27:38]
        # distributions[4] = batch_expert[:, 38:49]
        # distributions[5] = batch_expert[:, 49:60]
        # distributions[6] = batch_expert[:, 60:62]
        # distributions[7] = batch_expert[:, 62:64]
        # distributions[8] = batch_expert[:, 64:67]
        # distributions[9] = batch_expert[:, 67:85]
        # distributions[10] = batch_expert[:, 85:88]
        features_num = 30
        distributions = [None] * features_num
        distributions[0] = batch_expert[:, 0:9]
        distributions[1] = batch_expert[:, 9:10]
        distributions[2] = batch_expert[:, 10:12]
        distributions[3] = batch_expert[:, 12:14]
        distributions[4] = batch_expert[:, 14:15]
        distributions[5] = batch_expert[:, 15:23]
        distributions[6] = batch_expert[:, 23:24]
        distributions[7] = batch_expert[:, 24:33]
        distributions[8] = batch_expert[:, 33:34]
        distributions[9] = batch_expert[:, 34:41]
        distributions[10] = batch_expert[:, 41:42]
        distributions[11] = batch_expert[:, 42:50]
        distributions[12] = batch_expert[:, 50:52]
        distributions[13] = batch_expert[:, 52:59]
        distributions[14] = batch_expert[:, 59:60]
        distributions[15] = batch_expert[:, 60:61]
        distributions[16] = batch_expert[:, 61:77]
        distributions[17] = batch_expert[:, 77:78]
        distributions[18] = batch_expert[:, 78:81]
        distributions[19] = batch_expert[:, 81:82]
        distributions[20] = batch_expert[:, 82:83]
        distributions[21] = batch_expert[:, 83:90]
        distributions[22] = batch_expert[:, 90:95]
        distributions[23] = batch_expert[:, 95:100]
        distributions[24] = batch_expert[:, 100:103]
        distributions[25] = batch_expert[:, 103:106]
        distributions[26] = batch_expert[:, 106:109]
        distributions[27] = batch_expert[:, 109:112]
        distributions[28] = batch_expert[:, 112:115]
        distributions[29] = batch_expert[:, 115:118]

        kl = 0.0
        for i in range(features_num):
            kl += (F.softmax(batch_gen_feature[i].to(device), dim=1) *
                   (F.log_softmax(batch_gen_feature[i].to(device), dim=1) -
                    F.log_softmax(distributions[i].to(device), dim=1))).sum(dim=1).mean()

        return kl

    def save_model(self):
        torch.save(self.G.state_dict(), r'./model/user_G.pt')
        torch.save(self.D.state_dict(), r'./model/user_D.pt')
