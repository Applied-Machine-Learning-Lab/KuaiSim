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

class KREnvironment_WholeSession_TemperDiscount(KREnvironment_WholeSession_GPU):
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
    - no user leave model: see self.get_leave_signal
    - candidate item pool: self.candidate_ids, self.candidate_item_meta
    - history monitor: self.env_history, not set up until self.reset
    '''
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - temper_discount
        - from KREnvironment_WholeSession_GPU:
            - uirm_log_path
            - slate_size
            - episode_batch_size
            - item_correlation
            - from BaseRLEnvironment
                - max_step_per_episode
                - initial_temper
        '''
        parser = KREnvironment_WholeSession_GPU.parse_model_args(parser)
        parser.add_argument('--temper_discount', type=float, default=2.0, 
                            help='how quickly the temper drops by scale if the recommendation is bad')
        return parser
    
    def __init__(self, args):
        '''
        from BaseRLEnvironment:
            self.max_step_per_episode
            self.initial_temper
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
        '''
        self.temper_discount = args.temper_discount
        super(KREnvironment_WholeSession_TemperDiscount, self).__init__(args)
        
    
    def get_leave_signal(self, user_state, action, response_dict):
        '''
        User leave model maintains the user temper, and a user leaves when the temper drops below 1.
        @input:
        - user_state: not used in this env
        - action: not used in this env
        - response_dict: (B, slate_size, n_feedback)
        @process:
        - update temper
        @output:
        - done_mask: 
        '''
        # (B, slate_size, n_feedback)
        point_reward = response_dict['immediate_response'] * self.response_weights.view(1,1,-1)
        # (B, slate_size)
        combined_reward = torch.sum(point_reward, dim = 2)
        # (B, )
        temper_boost = torch.mean(combined_reward, dim = 1)
        # temper update for leave model
        temper_update = (temper_boost - 2) * self.temper_discount
        temper_update[temper_update > 0] = 0
#         temper_update[temper_update < -2] = -2
        self.current_temper += temper_update
        # leave signal
        done_mask = self.current_temper < 1
        return done_mask
        