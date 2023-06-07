import numpy as np
import torch
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import torch.nn.functional as F

import utils
from reader import *
from model.simulator import *
from env.KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU

class KREnvironment_SlateRec(KREnvironment_WholeSession_GPU):
    '''
    KuaiRand simulated environment for list-wise recommendation
    Main interface:
    - parse_model_args: for hyperparameters
    - reset: reset online environment, monitor, and corresponding initial observation
    - step: action --> new observation, user feedbacks, and other updated information
    - get_candidate_info: obtain the entire item candidate pool
    Main Components:
    - data reader: self.reader for user profile&history sampler
    - user immediate response model: see self.get_response
    - user leave model: not used in this env, see self.get_leave_signal
    - candidate item pool: self.candidate_ids, self.candidate_item_meta
    - history monitor: self.env_history, not set up until self.reset
    '''
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
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
        return parser
    
    def __init__(self, args):
        '''
        from KREnvironment_WholeSession_GPU:
            self.uirm_log_path
            self.slate_size
            self.rho
            self.immediate_response_stats: reader statistics for user response model
            self.immediate_response_model: the ground truth user response model
            self.max_hist_len
            self.response_types
            self.response_dim: number of feedback_type
            self.response_weights
            self.reader
            self.candidate_iids: [encoded item id]
            self.candidate_item_meta: {'if_{feature_name}': (n_item, feature_dim)}
            self.n_candidate
            self.candidate_item_encoding: (n_item, item_enc_dim)
            self.gt_state_dim: ground truth user state vector dimension
            self.action_dim: slate size
            self.observation_space: see reader.get_statistics()
            self.action_space: n_condidate
            from BaseRLEnvironment:
                self.max_step_per_episode
                self.initial_temper
        '''
        super(KREnvironment_SlateRec, self).__init__(args)
        
        self.response_weights = torch.tensor([0 if f == 'is_hate' else 1 \
                                              for f in self.response_types]).to(torch.float).to(args.device)
        if args.single_response:
            self.response_weights = torch.zeros_like(self.response_weights)
            self.response_weights[0] = 1
            
    def reset(self, params = {'empty_history': True}):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': scalar, 
                   'empty_history': True if start from empty history, 
                   'initial_history': start with initial history}
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

        # batch-wise monitor
        self.env_history = {'coverage': [], 'ILD': []}
        
        return deepcopy(self.current_observation)
    
    def step(self, step_dict, update_observation = True):
        '''
        users react to the recommendation action
        @input:
        - step_dict: {'action': (B, slate_size),
                      'action_features': (B, slate_size, item_dim) }
        @output:
        - new_observation: {'user_profile': {'user_id': (B,), 
                                             'uf_{feature_name}': (B, feature_dim)}, 
                            'user_history': {'history': (B, max_H), 
                                             'history_if_{feature_name}': (B, max_H, feature_dim), 
                                             'history_{response}': (B, max_H), 
                                             'history_length': (B, )}}
        - response_dict: {'immediate_response': see self.get_response@output - response_dict,
                          'done': (B,)}
        - update_info: see self.update_observation@output - update_info
        '''
        
        # URM forward
        with torch.no_grad():
            action = step_dict['action'] # must be indices on candidate_ids
            
            # get user response
            response_dict = self.get_response(step_dict)
            response = response_dict['immediate_response']

            # done mask and temper update
            # (B,)
            done_mask = self.get_leave_signal(None, None, response_dict, update_observation) # this will also change self.current_temper
            response_dict['done'] = done_mask
            
            # update user history in current_observation
            # {'slate': (B, slate_size), 'updated_observation': a copy of self.current_observation}
            update_info = self.update_observation(None, action, response, done_mask, update_observation)
            
            # env_history update: step, leave, temper, converage, ILD
            if update_observation:
                self.current_step += 1
                n_leave = done_mask.sum()
                self.env_history['coverage'].append(response_dict['coverage'])
                self.env_history['ILD'].append(response_dict['ILD'])

                # when users left, new users come into the running batch
                if done_mask.sum() == len(done_mask):
                    sample_info = self.sample_new_batch_from_reader()
                    self.current_observation = self.get_observation_from_batch(sample_info)
                    self.current_step *= 0
                    self.current_temper *= 0
                    self.current_temper += self.initial_temper
                elif done_mask.sum() > 0:
                    print(done_mask)
                    print("User leave not synchronized")
                    raise NotImplemented

        return deepcopy(self.current_observation), response_dict, update_info

    
    def get_leave_signal(self, user_state, action, response_dict, update_temper = True):
        '''
        User leave model maintains the user temper, and a user leaves when the temper drops below 1.
        @input:
        - user_state: not used in this env
        - action: not used in this env
        - response_dict: {'immediate_response': (B, slate_size, n_feedback), ...(other unrelated info)} 
        @process:
        - update temper
        @output:
        - done_mask: 
        '''
        temper_down = 1
        new_temper = self.current_temper - temper_down
        done_mask = new_temper < 1
        if update_temper:
            self.current_temper = new_temper
        return done_mask

        