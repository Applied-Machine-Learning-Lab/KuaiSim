import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.DDPG import DDPG
    
class PGRA(DDPG):
    @staticmethod
    def parse_model_args(parser):
        '''
        - inverse_lr
        - inverse_decay
        - args from DDPG:
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
        parser = DDPG.parse_model_args(parser)
        parser.add_argument('--inverse_lr', type=float, default=0.0001, 
                            help='inverse module loss coefficient')
        parser.add_argument('--inverse_decay', type=float, default=0.00003, 
                            help='inverse module loss coefficient')
        return parser
    
    
    def __init__(self, args, facade):
        '''
        from DDPG:
            self.episode_batch_size
            self.batch_size
            self.actor_lr
            self.critic_lr
            self.actor_decay
            self.critic_decay
            self.actor
            self.actor_target
            self.actor_optimizer
            self.critic
            self.critic_target
            self.critic_optimizer
            self.tau
            from BaseRLAgent:
                self.gamma
                self.n_iter
                self.train_every_n_step
                self.check_episode
                self.with_eval
                self.save_path
                self.facade
                self.exploration_scheduler
        '''
        super().__init__(args, facade)
        self.inverse_model = facade.inverse_model
        self.inverse_lr = args.inverse_lr
        self.inverse_decay = args.inverse_decay
        self.inverse_model_optimizer = torch.optim.Adam([{'params': self.inverse_model.parameters()}, 
                                                         {'params': self.actor.get_scorer_parameters()}], 
                                                         lr=args.inverse_lr, weight_decay=args.inverse_decay)

                                    
    def action_before_train(self):
        super().action_before_train()
        self.training_history["inverse_loss"] = []

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
        
        critic_loss, actor_loss, inverse_loss = self.get_pgra_loss(observation, policy_output, reward, done_mask, next_observation)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['inverse_loss'].append(inverse_loss.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic_loss'][-1], 
                              self.training_history['inverse_loss'][-1])}
    
    def get_pgra_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                      do_actor_update = True, do_critic_update = True):
        
        critic_loss, actor_loss = self.get_ddpg_loss(observation, policy_output, reward, done_mask, next_observation,
                                                     do_actor_update = do_actor_update, do_critic_update = do_critic_update)
            
        # inverse module loss
        
        inferred_action = self.facade.infer_latent(observation, next_observation, self.actor)
        # (B,L)
        candidate_score = inferred_action['candidate_score']
        # (B,K)
        target_action = policy_output['action']
        # (B,K)
        target_score = torch.gather(candidate_score, 1, target_action-1)
        inverse_loss = (-torch.log(target_score)).mean()
        
        if do_actor_update and self.inverse_lr > 0:
            self.inverse_model_optimizer.zero_grad()
            # Optimize the actor 
            inverse_loss.backward()
            self.inverse_model_optimizer.step()
            
        return critic_loss, actor_loss, inverse_loss
