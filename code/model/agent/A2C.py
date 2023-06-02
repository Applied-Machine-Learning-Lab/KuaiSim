import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.BaseRLAgent import BaseRLAgent
    
class A2C(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - episode_batch_size
        - batch_size
        - actor_lr
        - critic_lr
        - actor_decay
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - gamma
            - n_iter
            - train_every_n_step
            - initial_greedy_epsilon
            - final_greedy_epsilon
            - elbow_greedy
            - check_episode
            - with_eval
            - save_path
        '''
        parser = BaseRLAgent.parse_model_args(parser)
        # parser.add_argument('--episode_batch_size', type=int, default=8, 
        #                     help='episode sample batch size')
        # parser.add_argument('--batch_size', type=int, default=32, 
        #                     help='training batch size')
        # parser.add_argument('--actor_lr', type=float, default=1e-4, 
        #                     help='learning rate for actor')
        parser.add_argument('--critic_lr', type=float, default=1e-4, 
                            help='decay rate for critic')
        # parser.add_argument('--actor_decay', type=float, default=1e-4, 
        #                     help='learning rate for actor')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
                            help='mitigation factor')
        parser.add_argument('--advantage_bias', type=float, default=0, 
                            help='mitigation factor')
        parser.add_argument('--entropy_coef', type=float, default=0.1, 
                            help='mitigation factor')
        return parser
    
    
    def __init__(self, *input_args):
        '''
        self.gamma
        self.n_iter
        self.check_episode
        self.with_eval
        self.save_path
        self.facade
        self.exploration_scheduler
        '''
        args, env, actor, critic, buffer = input_args
        super().__init__(args, env, actor, buffer)
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size
        
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.actor_decay = args.actor_decay
        self.critic_decay = args.critic_decay
        
        # self.actor = facade.actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)

        self.tau = args.target_mitigate_coef
        self.advantage_bias = args.advantage_bias
        self.entropy_coef = args.entropy_coef
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
        
        
#     def action_after_train(self):
#         self.facade.stop_env()
        
#     def get_report(self):
#         episode_report = self.facade.get_episode_report(10)
#         train_report = {k: np.mean(v[-10:]) for k,v in self.training_history.items()}
#         return episode_report, train_report
        
    def action_before_train(self):
        super().action_before_train()
        # self.training_history['entropy_loss'] = []
        # self.training_history['advantage'] = []
        self.training_history = {'actor_loss': [], 'critic_loss': [], 'entropy_loss':[], 'advantage':[],
                            'Q': [], 'next_Q': []}
        
    # def run_episode_step(self, *episode_args):
    #     '''
    #     One step of interaction
    #     '''
    #     episode_iter, epsilon, observation, do_buffer_update = episode_args
    #     with torch.no_grad():
    #         # sample action
    #         policy_output = self.facade.apply_policy(observation, self.actor, epsilon, 
    #                                                  do_explore = True, do_softmax = True)
    #         # apply action on environment and update replay buffer
    #         next_observation, reward, done, info = self.facade.env_step(policy_output)
    #         # update replay buffer
    #         if do_buffer_update:
    #             self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
    #     return next_observation
            

    def step_train(self):
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)
        reward = user_feedback['reward'].view(-1)
#         reward = torch.FloatTensor(reward)
#         done_mask = torch.FloatTensor(done_mask)
        
        critic_loss, actor_loss, entropy_loss, advantage = self.get_a2c_loss(observation, policy_output, reward, done_mask, next_observation)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['entropy_loss'].append(entropy_loss.item())
        self.training_history['advantage'].append(advantage.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic_loss'][-1], 
                              self.training_history['entropy_loss'][-1], 
                              self.training_history['advantage'][-1])}
    
    def get_a2c_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                      do_actor_update = True, do_critic_update = True):
        
        # Get current Q estimate
        current_policy_output = self.apply_policy(observation, self.actor)
        current_target_critic_output = self.apply_critic(observation, current_policy_output, self.critic_target)
        V_S = current_target_critic_output['v']
        
        # Compute the target Q value
        next_policy_output = self.apply_policy(next_observation, self.actor_target)
#         next_policy_output = self.facade.apply_p[olicy(next_observation, self.actor)
        target_critic_output = self.apply_critic(next_observation, next_policy_output, self.critic_target)
        V_S_prime = target_critic_output['v'].detach()
        # S_prime = next_policy_output['state_emb']
        # V_S_prime = self.critic_target({'state_emb': S_prime})['v'].detach()
#         V_S_prime = self.critic({'state_emb': S_prime})['v'].detach()
        
        Q_S = reward + self.gamma * (done_mask * V_S_prime)
        advantage = torch.clamp((Q_S - V_S).detach(), -1, 1) # (B,)

        # Compute critic loss
        value_loss = F.mse_loss(V_S, Q_S).mean()
        
        # Regularization loss
#         critic_reg = current_critic_output['reg']

        if do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        # Compute actor loss
        current_policy_output = self.apply_policy(observation, self.actor)
        A = policy_output['action']
#         logp = -torch.log(current_policy_output['action_prob'] + 1e-6) # (B,K)
        logp = -torch.log(current_policy_output['probs'] + 1e-6) # (B,K)
        # use log(1-p), p is close to zero when there are large number of items
#         logp = torch.log(-torch.gather(current_policy_output['candidate_prob'],1,A-1)+1) # (B,K)
        actor_loss = torch.mean(torch.sum(logp * (advantage.view(-1,1) + self.advantage_bias), dim = 1))
        entropy_loss = torch.sum(current_policy_output['all_probs'] \
                                  * torch.log(current_policy_output['all_probs']), dim = 1).mean()
        
        # Regularization loss
#         actor_reg = policy_output['reg']

        if do_actor_update and self.actor_lr > 0:
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            (actor_loss + self.entropy_coef * entropy_loss).backward()
            self.actor_optimizer.step()
            
        return value_loss, actor_loss, entropy_loss, torch.mean(advantage)

    def apply_policy(self, observation, policy_model, epsilon = 0, 
                     do_explore = False, do_softmax = True):
        '''
        @input:
        - observation: input of policy model
        - policy_model
        - epsilon: greedy epsilon, effective only when do_explore == True
        - do_explore: exploration flag, True if adding noise to action
        - do_softmax: output softmax score
        '''
#         feed_dict = utils.wrap_batch(observation, device = self.device)
        feed_dict = observation
        # out_dict = policy_model(feed_dict)
        is_train = True
        input_dict = {'observation': observation, 
                'candidates': self.env.get_candidate_info(observation), 
                'epsilon': epsilon, 
                'do_explore': do_explore, 
                'is_train': is_train, 
                'batch_wise': False}
        out_dict = policy_model(input_dict)
#         if do_explore:
#             action_emb = out_dict['action']
#             # sampling noise of action embedding
#             if np.random.rand() < epsilon:
#                 action_emb = torch.clamp(torch.rand_like(action_emb)*self.noise_var, -1, 1)
#             else:
#                 action_emb = action_emb + torch.clamp(torch.rand_like(action_emb)*self.noise_var, -1, 1)
# #                 self.noise_var -= self.noise_decay
#             out_dict['action'] = action_emb
            
        # if 'candidate_ids' in feed_dict:
        #     # (B, L, item_dim)
        #     out_dict['candidate_features'] = feed_dict['candidate_features']
        #     # (B, L)
        #     out_dict['candidate_ids'] = feed_dict['candidate_ids']
        #     batch_wise = True
        # else:
        #     # (1,L,item_dim)
        #     out_dict['candidate_features'] = self.candidate_features.unsqueeze(0)
        #     # (L,)
        #     out_dict['candidate_ids'] = self.candidate_iids
        #     batch_wise = False
            
        # # action prob (B,L)
        # action_prob = policy_model.score(out_dict['action_emb'], 
        #                                  out_dict['candidate_features'], 
        #                                  do_softmax = do_softmax)

        # # two types of greedy selection
        # if np.random.rand() >= self.topk_rate:
        #     # greedy random: categorical sampling
        #     action, indices = utils.sample_categorical_action(action_prob, out_dict['candidate_ids'], 
        #                                                       self.slate_size, with_replacement = False, 
        #                                                       batch_wise = batch_wise, return_idx = True)
        # else:
        #     # indices on action_prob
        #     _, indices = torch.topk(action_prob, k = self.slate_size, dim = 1)
        #     # topk action
        #     if batch_wise:
        #         action = torch.gather(out_dict['candidate_ids'], 1, indices).detach() # (B, slate_size)
        #     else:
        #         action = out_dict['candidate_ids'][indices].detach() # (B, slate_size)
        # # (B,K)
        # out_dict['action'] = action 
        # # (B,K,item_dim)
        # out_dict['action_features'] = self.candidate_features[action-1]
        # # (B,K)
        # out_dict['action_prob'] = torch.gather(action_prob, 1, indices) 
        # # (B,L)
        # out_dict['candidate_prob'] = action_prob
        return out_dict
        
    def apply_critic(self, observation, policy_output, critic_model):
        # feed_dict = {"state_emb": policy_output["state_emb"], 
        #              "action_emb": policy_output["action_emb"]}
        feed_dict = {'state': policy_output['state'],
                'action': policy_output['hyper_action']}
        critic_output = critic_model(feed_dict)
        return critic_output 

    def save(self):
        torch.save(self.critic.state_dict(), self.save_path + "_critic")
        torch.save(self.critic_optimizer.state_dict(), self.save_path + "_critic_optimizer")

        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")


    def load(self):
        self.critic.load_state_dict(torch.load(self.save_path + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(self.save_path + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
