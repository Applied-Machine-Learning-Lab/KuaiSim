import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.BaseRLAgent import BaseRLAgent
    
class DDPG(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_lr
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - gamma
            - reward_func
            - n_iter
            - train_every_n_step
            - start_policy_train_at_step
            - initial_epsilon
            - final_epsilon
            - elbow_epsilon
            - explore_rate
            - do_explore_in_train
            - check_episode
            - save_episode
            - save_path
            - actor_lr
            - actor_decay
            - batch_size
        '''
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--critic_lr', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
                            help='mitigation factor')
        return parser
    
    
    def __init__(self, *input_args):
        '''
        components:
        - critic
        - critic_optimizer
        - actor_target
        - critic_target
        - components from BaseRLAgent:
            - env
            - actor
            - actor_optimizer
            - buffer
            - exploration_scheduler
            - registered_models
        '''
        args, env, actor, critic, buffer = input_args
        super().__init__(args, env, actor, buffer)
        
        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.tau = args.target_mitigate_coef
        
        # models
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        
        # controller
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)
        self.do_actor_update = True
        self.do_critic_update = True

        # register models that will be saved
        self.registered_models.append((self.critic, self.critic_optimizer, "_critic"))
        
        
    def action_before_train(self):
        '''
        Action before training:
        - buffer setup
        - monitor setup
        - run random episodes to build-up the initial buffer
        '''
        super().action_before_train()
        
        # training records
        self.training_history = {'actor_loss': [], 'critic_loss': [], 
                                 'Q': [], 'next_Q': []}
        

    def step_train(self):
        '''
        @process:
        - buffer.sample(): batch_size --> observation, policy_output, user_response, done_mask, next_observation
            - observation: see self.env.step@output - new_observation
            - policy_output: {
                'state': (B,state_dim), 
                'action': (B,K),
                ...}
            - user_feedback: {
                'reward': (B,),
                'immediate_response': (B,K*n_feedback)}
            - done_mask
            - next_observation
        - policy.get_forward(): observation, candidates --> policy_output
        - policy.get_loss(): observation, candidates, policy_output, user_response --> loss
        - optimizer.zero_grad(); loss.backward(); optimizer.step()
        - update training history
        '''
        
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)
        epsilon = 0
        is_train = True
        # (B, )
        reward = user_feedback['reward'].view(-1)
        
        # DDPG loss
        
        # Get current Q estimate
        current_critic_output = self.apply_critic(observation, policy_output, self.critic)
        # (B, )
        current_Q = current_critic_output['q']
        
        # Compute the target Q value
        next_policy_output = self.apply_policy(next_observation, self.actor_target, 
                                               0., False, is_train)
        target_critic_output = self.apply_critic(next_observation, next_policy_output, self.critic_target)
        next_Q = target_critic_output['q']
        # (B, )
        target_Q = reward + self.gamma * (done_mask * next_Q).detach()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q).mean()
        
        # Regularization loss
#         critic_reg = current_critic_output['reg']

        if self.do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Compute actor loss
        policy_output = self.apply_policy(observation, self.actor, 
                                          0., self.do_explore_in_train, is_train)
        critic_output = self.apply_critic(observation, policy_output, self.critic)
        actor_loss = -critic_output['q'].mean()
        
        # Regularization loss
#         actor_reg = policy_output['reg']

        if self.do_actor_update and self.actor_lr > 0:
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        loss_dict = {'actor_loss': actor_loss.item(), 
                     'critic_loss': critic_loss.item(), 
                     'Q': torch.mean(current_Q).item(), 
                     'next_Q': torch.mean(next_Q).item()}
    
        for k in loss_dict:
            if k in self.training_history:
                try:
                    self.training_history[k].append(loss_dict[k].item())
                except:
                    self.training_history[k].append(loss_dict[k])
        
        return loss_dict
    
    def apply_policy(self, observation, actor, *policy_args):
        '''
        @input:
        - observation:{'user_profile':{
                           'user_id': (B,)
                           'uf_{feature_name}': (B,feature_dim), the user features}
                       'user_history':{
                           'history': (B,max_H)
                           'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
        - actor: the actor model
        - epsilon: scalar
        - do_explore: boolean
        - is_train: boolean
        @output:
        - policy_output
        '''
        epsilon = policy_args[0]
        do_explore = policy_args[1]
        is_train = policy_args[2]
        input_dict = {'observation': observation, 
                      'candidates': self.env.get_candidate_info(observation), 
                      'epsilon': epsilon, 
                      'do_explore': do_explore, 
                      'is_train': is_train, 
                      'batch_wise': False}
        out_dict = self.actor(input_dict)
        return out_dict
    
    def apply_critic(self, observation, policy_output, critic):
        feed_dict = {'state': policy_output['state'],
                     'action': policy_output['hyper_action']}
        return self.critic(feed_dict)

    def save(self):
        super().save()

    def load(self):
        super().load()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
