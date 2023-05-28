import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.A2C import A2C
    
class BehaviorA2C(A2C):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - behavior_lr
        - behavior_decay
        - args from A2C:
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
        parser = A2C.parse_model_args(parser)
        parser.add_argument('--behavior_lr', type=float, default=0.0001, 
                            help='behaviorvise loss coefficient')
        parser.add_argument('--behavior_decay', type=float, default=0.00003, 
                            help='behaviorvise loss coefficient')
        return parser
    
    
    def __init__(self, args, facade):
        '''
        self.gamma
        self.n_iter
        self.check_episode
        self.with_eval
        self.save_path
        self.facade
        self.exploration_scheduler
        '''
        super().__init__(args, facade)
        self.behavior_lr = args.behavior_lr
        self.behavior_decay = args.behavior_decay
        self.actor_behavior_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.behavior_lr, 
                                                      weight_decay=args.behavior_decay)
        
        
#     def action_after_train(self):
#         self.facade.stop_env()
        
#     def get_report(self):
#         episode_report = self.facade.get_episode_report(10)
#         train_report = {k: np.mean(v[-10:]) for k,v in self.training_history.items()}
#         return episode_report, train_report
        
    def get_behavior_loss(self, observation, policy_output, next_observation, do_update = True):
#         observation, exposure, feedback = self.facade.sample_supervise_data(self.batch_size)
        observation, exposure, feedback = self.facade.extract_behavior_data(observation, policy_output, next_observation)
        observation['candidate_ids'] = exposure['ids']
        observation['candidate_features'] = exposure['features']
        policy_output = self.facade.apply_policy(observation, self.actor, do_softmax = False)
        action_prob = torch.sigmoid(policy_output['candidate_prob'])
        behavior_loss = F.binary_cross_entropy(action_prob, feedback)
#         return behavior_loss
#         behavior_loss = self.facade.get_policy_gradient_loss(observation, policy_output, next_observation, self.actor)
        
        if do_update and self.behavior_lr > 0:
            self.actor_behavior_optimizer.zero_grad()
            behavior_loss.backward()
            self.actor_behavior_optimizer.step()
        return behavior_loss

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
#         reward = torch.FloatTensor(reward)
#         done_mask = torch.FloatTensor(done_mask)
        
        critic_loss, actor_loss = self.get_a2c_loss(observation, policy_output, reward, done_mask, next_observation)
        behavior_loss = self.get_behavior_loss(observation, policy_output, next_observation)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['behavior_loss'].append(behavior_loss.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic_loss'][-1], 
                              self.training_history['behavior_loss'][-1])}
            