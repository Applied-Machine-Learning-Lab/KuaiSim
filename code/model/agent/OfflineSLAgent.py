import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.BaseRLAgent import BaseRLAgent
    
class OfflineSLAgent(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - episode_batch_size
        - batch_size
        - actor_lr
        - actor_decay
        - args from BaseRLAgent:
            - gamma
            - n_iter: equivalent to n_epoch in BaseSLAgent
            - train_every_n_step
            - initial_greedy_epsilon
            - final_greedy_epsilon
            - elbow_greedy
            - check_episode
            - with_eval
            - save_path
        '''
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--episode_batch_size', type=int, default=8, 
                            help='episode sample batch size')
        parser.add_argument('--batch_size', type=int, default=32, 
                            help='training batch size')
        parser.add_argument('--actor_lr', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--actor_decay', type=float, default=1e-4, 
                            help='learning rate for actor')
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
        self.device = args.device
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size
        self.n_epoch = self.n_iter
        
        self.actor_lr = args.actor_lr
        self.actor_decay = args.actor_decay
        
        self.actor = facade.actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)

        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
        
    def action_before_train(self):
        '''
        Action before training:
        - setup training_history recorder
        '''
        # training records
        self.train_iterator = self.facade.env.get_new_iterator(self.batch_size)
        self.training_history = {"training_loss": []}
        self.facade.buffer_head = 0
        self.facade.current_buffer_size = 0
        self.facade.n_stream_record = 0
        self.facade.is_training_available = False
        
    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            # sample action
            policy_output = self.facade.apply_policy(observation, self.actor, 0, do_explore = False)
            # apply action on environment and update replay buffer
            next_observation, reward, done, info = self.facade.env_step(policy_output)
        return next_observation
            
    def step_train(self):
        try:
            batch = next(self.train_iterator)
            if batch['exposure'].shape[0] != self.batch_size:
                self.train_iterator = self.facade.env.get_new_iterator(self.batch_size)
                batch = next(self.train_iterator)
        except:
            self.train_iterator = self.facade.env.get_new_iterator(self.batch_size)
            batch = next(self.train_iterator)
            
        wrapped_batch = utils.wrap_batch(batch, device = self.device)
#         negative_samples = self.sample_negative(wrapped_batch['exposure'])
        wrapped_batch['candidate_ids'] = wrapped_batch['exposure']
        wrapped_batch['candidate_features'] = wrapped_batch['exposure_features']
        policy_output = self.facade.apply_policy(wrapped_batch, self.actor, do_softmax = False)
        action_prob = torch.sigmoid(policy_output['candidate_prob'])
        supervise_loss = F.binary_cross_entropy(action_prob, wrapped_batch['feedback'])
        self.training_history['training_loss'].append(torch.mean(supervise_loss).item())
        if self.actor_lr > 0:
            self.actor_optimizer.zero_grad()
            supervise_loss.backward()
            self.actor_optimizer.step()
        return supervise_loss

#     def sample_negative(self, positive):
        

    def save(self):
        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")


    def load(self):

        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
