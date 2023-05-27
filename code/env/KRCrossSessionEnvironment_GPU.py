import numpy as np
import utils
import torch
import torch.nn as nn
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from reader import *
from model.simulator import *
from env.KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU
import utils


class KRCrossSessionEnvironment_GPU(KREnvironment_WholeSession_GPU):
    '''
    KuaiRand simulated environment on GPU machines
    Components:
    - multi-behavior user response model: 
        - (user history, user profile) --> user_state
        - (user_state, item) --> feedbacks (e.g. click, long_view, like, ...)
    - user leave model:
        - user temper reduces to <1 and leave
        - user temper drops gradually through time and further drops when the user is unsatisfactory about a recommendation
    - user retention model:
        - [end of session user_state] --> user retention
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - max_n_session
        - max_return_day
        - next_day_return_bias
        - feedback_influence_on_return
        - from KREnvironment_WholeSession_GPU:
            - uirm_log_path
            - slate_size
            - episode_batch_size
            - item_correlation
            - single_response
            - from BaseRLEnvironment
                - max_step_per_episode
                - initial_temper
        '''
        parser = KREnvironment_WholeSession_GPU.parse_model_args(parser)
        parser.add_argument('--max_n_session', type=int, default=30, 
                            help='max number of sessions per user episode')
        parser.add_argument('--max_return_day', type=int, required=10, 
                            help='next day return probability, geometric distribution')
        parser.add_argument('--next_day_return_bias', type=float, required=0.75, 
                            help='next day return probability, geometric distribution')
        parser.add_argument('--feedback_influence_on_return', type=float, required=0., 
                            help='next day return probability, geometric distribution')
        return parser
    
    def __init__(self, args):
        self.max_n_session = args.max_n_session
        self.max_return_day = args.max_return_day
        self.next_day_return_bias = args.next_day_return_bias
        self.feedback_influence_on_return = args.feedback_influence_on_return
        assert not args.single_response
        assert 0. < args.next_day_return_bias <= 1.
        super().__init__(args)
        # global return day bias
        logP = np.log(self.next_day_return_bias)
        log1_P = np.log(1 - self.next_day_return_bias) if self.next_day_return_bias < 1 else -10.
        self.day_bias_multiplier = torch.FloatTensor([i for i in range(self.max_return_day)]).to(args.device)
        # random personal return day bias
        self.random_personalization = nn.Linear(self.gt_state_dim, 1).to(args.device)
        
        self.action_dim = self.immediate_response_model.feedback_dim
        
        
        
    def reset(self, params = {}):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': scalar, 
                    'empty_history': True if start from empty history, default = False
                    'initial_history': start with initial history, empty_history must be False}
        @process:
        - self.batch_iter
        - self.current_observation
        - self.current_step
        - self.current_temper
        - self.env_history
        @output:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H, feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
        '''
        super().reset(params)
        
        self.user_leave_history = []
        self.user_return_history = []
        self.user_total_return_gap = []
        self.env_history = {'return_day': [], 
                            'total_return_gap': []}

        self.session_count = 0
        return deepcopy(self.current_observation)
    
    
    def step(self, step_dict):
        '''
        @input:
        - step_dict: {'action': (B, W_dim)}
        '''
        # actions (exposures)
        action = step_dict['action'] # (B, W_dim), the fusion weight
        
        # user interaction
        with torch.no_grad():
            # point-wise model gives xtr scores
            # (B, 1, state_dim)
            profile_dict = {k:v for k,v in self.current_observation['user_profile'].items()}
            user_state = self.get_ground_truth_user_state(profile_dict,
                                            {k:v for k,v in self.current_observation['user_history'].items()})
            user_state = user_state.view(self.episode_batch_size, 1, self.gt_state_dim)
            # (B, n_item, n_feedback), _
            behavior_scores, _ = self.immediate_response_model.get_pointwise_scores(user_state, 
                                                    self.candidate_item_encoding[None,:,None,:], 
                                                    self.episode_batch_size)
            # generate recommendation list
            ########################################
            # This is where the action take effect #
            # (B, n_item, n_feedback)
            point_scores = torch.sigmoid(behavior_scores)
            # (B, n_item)
            ranking_scores = torch.sum(point_scores * action.view(self.episode_batch_size,1,self.action_dim), dim = -1)
            ########################################
            # _, (B, slate_size)
            _, indices = torch.topk(ranking_scores, self.slate_size, dim = -1)
            
            # sample immediate responses
            # (B, slate_size, n_feedback)
            score_indices = torch.tile(indices[:,:,None], (1,1,point_scores.shape[-1]))
            selected_scores = torch.gather(point_scores, 1, score_indices) 
            # (B, slate_size, n_feedback)
            response = torch.bernoulli(selected_scores)

            # get leave signal
            # (B,), 0-1 vector
            done_mask = self.get_leave_signal(response)
            self.user_leave_history.append(done_mask.detach().cpu().numpy())
            
            # update observation
            update_info = self.update_observation(user_state, indices, response, done_mask)
            
            

            #  all environment users leave at the same time
            if done_mask.sum() == len(done_mask):
                
                # get retention signal
                retention_out_dict = self.get_retention(update_info['updated_observation'], response, done_mask)

                # (B, max_return_day)
                retention_prob = torch.softmax(retention_out_dict['preds'], dim = -1)
                return_day = Categorical(retention_prob).sample() + 1
                return_day = return_day.to(torch.float)
                self.user_return_history.append(return_day.detach().cpu().numpy())
                self.env_history['return_day'].append(torch.mean(return_day).item())
                
                # refresh temper in every session
                self.session_count += 1
                # refresh users when the last session is done 
                if self.session_count == self.max_n_session:
                    sample_info = self.sample_new_batch_from_reader()
                    new_observation = self.get_observation_from_batch(sample_info)
                    self.current_observation = new_observation
                    self.user_total_return_gap.append(np.sum(np.array(self.user_return_history[-self.max_n_session:]), axis = 0))
                    self.env_history['total_return_gap'].append(np.sum(self.env_history['return_day'][-self.max_n_session:]).item())
                    self.session_count = 0
                self.current_temper *= 0
                self.current_temper += self.initial_temper
            elif done_mask.sum() > 0:
                print(done_mask)
                print("Leaving not synchronized")
                raise NotImplemented
            else:
                return_day = torch.zeros(self.episode_batch_size).to(self.device)
        user_feedback = {'immediate_response': response, 
                         'user_state': user_state,
                         'done': done_mask, 
                         'retention': return_day.to(torch.float)}
        return deepcopy(self.current_observation), user_feedback, update_info


    def get_leave_signal(self, response):
        '''
        Make sure that users leave at the same time.
        @input:
        - user_state: (B, state_dim)
        - response: (B, slate_size, n_feedback)
        '''
        temper_down = 1
        self.current_temper -= temper_down
        done_mask = self.current_temper < 1
        return done_mask
    
    def get_retention(self, observation, response, done_mask):
        new_gt_state = self.get_ground_truth_user_state(observation['user_profile'], observation['user_history'])
        # (B, )
        personal_bias = self.random_personalization(new_gt_state).view(self.episode_batch_size)
        # (B, slate_size, n_feedback)
        point_reward = response * self.response_weights.view(1,1,-1)
        # (B, )
        response_bias = torch.mean(torch.sum(point_reward, dim=2), dim=1) * self.feedback_influence_on_return
        # (B, )
        P = torch.clamp(personal_bias + response_bias + self.next_day_return_bias, 0.001, 0.999)
        logP = torch.log(P)
        log1_P = torch.log(1-P)
        # (B, max_ret_day)
        preds = logP.view(-1, 1) + log1_P.view(-1,1) * self.day_bias_multiplier
        
        return {'preds': preds, 'state': new_gt_state}
    
    
        
        
        
        