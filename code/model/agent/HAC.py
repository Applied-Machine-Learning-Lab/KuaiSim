import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
from itertools import chain

import utils
from model.agent.DDPG import DDPG
from model.components import DNN
    
class HAC(DDPG):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - behavior_lr
        - behavior_decay
        - hyper_actor_coef
        - hyper_noise_std
        - hyper_noise_clip
        - from DDPG
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
        parser = DDPG.parse_model_args(parser)
        parser.add_argument('--behavior_lr', type=float, default=0.0001, 
                            help='behaviorvise loss coefficient')
        parser.add_argument('--behavior_decay', type=float, default=0.00003, 
                            help='behaviorvise loss coefficient')
        parser.add_argument('--hyper_actor_coef', type=float, default=0.1, 
                            help='hyper actor loss coefficient')
        return parser
    
    
    def __init__(self, *input_args):
        '''
        components:
        - actor_behavior_optimizer
        - components from DDPG:
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
        assert env.single_response
        super().__init__(*input_args)
        self.behavior_lr = args.behavior_lr
        self.behavior_decay = args.behavior_decay
        self.hyper_actor_coef = args.hyper_actor_coef
        
        self.inverse_module = DNN(actor.enc_dim, [256], actor.action_dim, 
                                  dropout_rate = actor.dropout_rate, do_batch_norm = True)
        self.inverse_module = self.inverse_module.to(args.device)
        self.inverse_module_optimizer = torch.optim.Adam(chain(self.inverse_module.parameters(), self.actor.parameters()), 
                                                         lr=args.hyper_actor_coef, weight_decay=args.actor_decay)
        self.actor_behavior_optimizer = torch.optim.Adam(self.actor.parameters(), 
                                                         lr=args.behavior_lr, weight_decay=args.behavior_decay)

    def setup_monitors(self):
        '''
        This is used in super().action_before_train() in super().train()
        Then the agent will call several rounds of run_episode_step() for collecting initial random data samples
        
        DDPG monitors:
        - training_history: actor_loss, critic_loss, Q, nextQ
        
        HAC extra monitors:
        - training_history: hyper_actor_loss, behavior_loss
        '''
        super().setup_monitors()
        self.training_history.update({"hyper_actor_loss": [], 
                                      "behavior_loss": []})
                                    

    def step_train(self):
        '''
        @process:
        - get sample
        - calculate Q'(s_{t+1}, a_{t+1}) and Q(s_t, a_t)
        - critic loss: TD error loss
        - critic optimization
        - actor loss: Q(s_t, \pi(s_t)) maximization
        - actor optimization
        '''
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)

        critic_loss, actor_loss, hyper_actor_loss, behavior_loss = self.get_hac_loss(observation, policy_output, user_feedback, done_mask, next_observation)
        
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['hyper_actor_loss'].append(hyper_actor_loss.item())
        self.training_history['behavior_loss'].append(behavior_loss.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic_loss'][-1], 
                              self.training_history['hyper_actor_loss'][-1], 
                              self.training_history['behavior_loss'][-1])}
    
    def get_hac_loss(self, observation, policy_output, user_feedback, 
                     done_mask, next_observation, 
                     do_actor_update = True, do_critic_update = True):
        
        epsilon = 0
        is_train = True
        # critic loss
        
        # current Q estimation: Q(s_t,a_t) = Q_Z(s_t, g(a_t))
        inverse_output = self.infer_hyper_action(observation, policy_output, self.actor)
        inverse_output.update({'state': policy_output['state'], 'action': inverse_output['Z']})
        current_critic_output = self.apply_critic(observation, inverse_output, self.critic)
        current_Q = current_critic_output['q'] # (B,)
        # target Q
        next_policy_output = self.apply_policy(next_observation, self.actor_target, epsilon, 
                                               False, is_train)
        target_critic_output = self.apply_critic(next_observation, next_policy_output, self.critic_target)
        next_Q = target_critic_output['q'] # (B,)
        reward = user_feedback['reward'].view(-1) # (B,)
        target_Q = reward + self.gamma * (done_mask * next_Q).detach()
        # compute TD error loss
        critic_loss = F.mse_loss(current_Q, target_Q).mean()
        if do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # actor loss
        
        # compute actor loss
        if do_actor_update and self.actor_lr > 0:
            self.actor_optimizer.zero_grad()
            temp_policy_output = self.apply_policy(observation, self.actor, 
                                                   epsilon, self.do_explore_in_train, is_train)
            critic_output = self.apply_critic(observation, temp_policy_output, self.critic)
            actor_loss = -critic_output['q'].mean()
            # Optimize the actor 
            actor_loss.backward()
            self.actor_optimizer.step()
            
        # hyper actor loss
        
        if do_actor_update and self.hyper_actor_coef > 0:
            self.inverse_module_optimizer.zero_grad()
            temp_policy_output = self.apply_policy(observation, self.actor, 
                                                   epsilon, True, is_train)
            inverse_output = self.infer_hyper_action(observation, temp_policy_output, self.actor)
            hyper_actor_loss = self.hyper_actor_coef * F.mse_loss(inverse_output['Z'], temp_policy_output['hyper_action']).mean()
            # Optimize the actor 
            hyper_actor_loss.backward()
            self.inverse_module_optimizer.step()
            
        # pointwise supervised loss
#         behavior_loss = self.get_behavior_loss(observation, policy_output, next_observation)
        if do_actor_update and self.behavior_lr > 0:
            # (B, K)
            A = policy_output['effect_action']
            # (B, K)
            point_label = user_feedback['immediate_response'].view(self.batch_size, self.env.slate_size, self.env.response_dim)[:,:,0]
            
            self.actor_behavior_optimizer.zero_grad()
            temp_policy_output = self.apply_policy(observation, self.actor, 
                                                   epsilon, False, is_train)
            candidate_scores = torch.gather(temp_policy_output['all_preds'], 1, A-1)
            action_prob = torch.sigmoid(candidate_scores)
            behavior_loss = F.binary_cross_entropy(action_prob, point_label).mean()
        
            behavior_loss.backward()
            self.actor_behavior_optimizer.step()
            
        return critic_loss, actor_loss, hyper_actor_loss, behavior_loss
    
    def infer_hyper_action(self, observation, policy_output, actor):
        '''
        inverse function or pooling for A --> Z
        @input:
        - observation
        - policy_output
        - actor
        @output:
        - Z_cap
        '''
        # (B,K)
        A = policy_output['effect_action'] 
        # (B,K,item_dim)
        item_info = self.env.get_candidate_info({'item_id': A}, all_item=False) 
        # (B,K,item_enc_dim)
        item_enc, reg = actor.user_encoder.get_item_encoding(
                            A, {k[3:]: v for k,v in item_info.items() if k != 'item_id'}, 
                            self.batch_size)
        item_enc = item_enc.view(self.batch_size, actor.effect_action_dim, actor.enc_dim)
        # (B,K,hyper_action_dim)
        itemwise_inverse = self.inverse_module(item_enc).view(self.batch_size, actor.effect_action_dim, actor.action_dim)
        # (B, hyper_action_dim)
        recovered_Z = torch.mean(itemwise_inverse, dim = 1).view(self.batch_size, actor.action_dim)
        return {'Z': recovered_Z}

        
    def apply_critic(self, observation, policy_output, critic_model):
        # feed_dict = {"state_emb": policy_output["state_emb"], 
        #              "action_emb": policy_output["action_emb"]}
        feed_dict = {'state': policy_output['state'],
                     'action': policy_output['action']}
        critic_output = critic_model(feed_dict)
        return critic_output  

    def save(self):
        super().save()

    def load(self):
        super().load()
 