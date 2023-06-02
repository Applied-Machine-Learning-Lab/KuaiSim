from MAIL.DiscriminatorModel import DiscriminatorModel
from MAIL.MailPolicy import MailPolicy
from MAIL.ValueModel import ValueModel
from algorithms.ppo import GAE, PPO_step
from utils.utils import *


class MailModel:
    def __init__(self, expert_data, lr_d=0.00001, lr_g=0.00003, lr_v=0.001, epochs=30, batch_size=512,
                 ppo_epoch=16, epsilon=0.1, l2_reg=1e-3):
        self.expert_data = expert_data

        self.epochs = epochs
        self.batch_size = batch_size
        self.ppo_epoch = ppo_epoch
        self.epsilon = epsilon
        self.l2_reg = l2_reg

        self.D = DiscriminatorModel()
        self.G = MailPolicy()
        self.V = ValueModel()

        self.optim_G = optim.Adam(self.G.parameters(), lr=lr_g)
        self.optim_D = optim.Adam(self.D.parameters(), lr=lr_d)
        self.optim_V = optim.Adam(self.V.parameters(), lr=lr_v)

        self.loss_func = nn.BCELoss()
        self.grad_map = {}

        to_device(self.G, self.D, self.V, self.loss_func)

    def train(self):
        writer = SummaryWriter()
        batch_num = (len(self.expert_data) + self.batch_size - 1) // self.batch_size
        iter_num = self.epochs * batch_num

        for epoch in range(self.epochs):
            # shuffle expert data
            idx = torch.randperm(len(self.expert_data))
            # generate (state, action) pairs
            expert_state_action_num = 10240
            self.G.generate_batch(expert_state_action_num)
            # sample generated (state, action) pairs from memory
            batch_gen_state, batch_gen_action, mask = self.G.sample_batch(512)

            ############################
            # update Discriminator
            ############################
            for i in range(batch_num):
                # sample a batch of expert trajectories
                batch_expert_old = [self.expert_data[int(k)] for k in
                                    idx[i * self.batch_size:(i + 1) * self.batch_size]]

                batch_expert = FLOAT([])
                for e in batch_expert_old:
                    batch_expert = torch.cat([batch_expert, e], dim=0)

                # batch_size = batch_expert.size(0)  # count (state, action) pairs

                # update Discriminator adaptively
                update_frequency = 1 if epoch < 10 else 20

                if i % update_frequency == 0:
                    self.optim_D.zero_grad()

                    expert_o = self.D(batch_expert.to(device))
                    gen_o = self.D(torch.cat([batch_gen_state, batch_gen_action.type(torch.float)], dim=1).to(device))

                    e_loss = self.loss_func(expert_o, torch.ones_like(expert_o, device=device))
                    g_loss = self.loss_func(gen_o, torch.zeros_like(gen_o, device=device))
                    d_loss = g_loss + e_loss

                    # g_loss = self.loss_func(gen_o, torch.ones_like(gen_o, device=device))
                    # e_loss = self.loss_func(expert_o, torch.zeros_like(expert_o, device=device))
                    # d_loss = -(g_loss + e_loss)

                    d_loss.register_hook(lambda grad: self.hook_grad("d_loss", grad))
                    d_loss.backward()

                    self.optim_D.step()

                writer.add_scalars('MAIL/Discriminator',
                                   {'Batch_D_loss': d_loss,
                                    'Batch_G_loss': g_loss,
                                    'Batch_E_loss': e_loss
                                    },
                                   epoch * batch_num + i)

                writer.add_scalars('MAIL/Reward',
                                   {'Batch_G_reward': gen_o.mean(),
                                    'Batch_E_reward': expert_o.mean()
                                    },
                                   epoch * batch_num + i)
                print("=" * 100 + "Discriminator")

                print(f'Epoch: {epoch}, Batch: {i}, Batch E loss: {e_loss.detach().cpu().numpy():.4f}, '
                      f'Batch G loss: {g_loss.cpu().detach().numpy(): .4f}, '
                      f'Batch D loss: {d_loss.cpu().detach().numpy(): .4f}, '
                      f'Batch G reward: {gen_o.mean().cpu().detach().numpy():.4f}, '
                      f'Batch E reward: {expert_o.mean().detach().cpu().numpy(): .4f}')

            with torch.no_grad():
                gen_r = self.D(torch.cat([batch_gen_state, batch_gen_action.type(torch.float)], dim=1).to(device))
                gen_r = gen_r.log()
                value_o = self.V(batch_gen_state)
                fixed_log_prob = self.G.get_log_prob(batch_gen_state, batch_gen_action)

            advantages, returns = GAE(- torch.log(1 - gen_r + 1e-6), mask, value_o, gamma=0.95, lam=0.95)

            ##############################
            # update Generator using PPO
            ##############################
            for k in range(15):
                new_index = torch.randperm(batch_gen_state.shape[0]).to(device)

                mini_batch_gen_state, mini_batch_gen_action, mini_batch_fixed_log_prob, mini_batch_returns, \
                mini_batch_advantages = batch_gen_state[new_index], batch_gen_action[new_index], fixed_log_prob[
                    new_index], returns[new_index], advantages[new_index]

                v_loss, p_loss = PPO_step(self.G, self.V, self.optim_G, self.optim_V, mini_batch_gen_state,
                                          mini_batch_gen_action,
                                          mini_batch_returns, mini_batch_advantages, mini_batch_fixed_log_prob,
                                          self.epsilon, self.l2_reg)

                writer.add_scalars('MAIL/Generator',
                                   {'Batch_V_loss': v_loss,
                                    'Batch_P_loss': p_loss
                                    },
                                   epoch * 15 + k)

                print("=" * 100 + "Generator")
                print(f'Epoch: {epoch}, Batch V loss: {v_loss.detach().cpu().numpy():.4f}, '
                      f'Batch P loss: {p_loss.detach().cpu().numpy(): .10f}'
                      )

            self.save_model()

            torch.cuda.empty_cache()

    def hook_grad(self, key, value):
        self.grad_map[key] = value

    def save_model(self):
        torch.save(self.G.state_dict(), r'./model/policy.pt')
        torch.save(self.V.state_dict(), r'./model/value.pt')
        torch.save(self.D.state_dict(), r'./model/discriminator.pt')


