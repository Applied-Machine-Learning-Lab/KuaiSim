import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from model.components import DNN
from utils import get_regularization
from model.general import BaseModel
from model.policy.BackboneUserEncoder import BackboneUserEncoder
    
class OneStagePolicy(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - from BackboneUserEncoder:
            - user_latent_dim
            - item_latent_dim
            - transformer_enc_dim
            - transformer_n_head
            - transformer_d_forward
            - transformer_n_layer
            - state_hidden_dims
            - state_dropout_rate
            - from BaseModel:
                - model_path
                - loss
                - l2_coef
        '''
        parser = BaseStateEncoder.parse_model_args(parser)
        return parser
    
    def __init__(self, args, env, device):
        self.slate_size = env.slate_size
        super().__init__(args, env.reader.get_statistics(), device)
        self.display_name = "OneStagePolicy"
        self.dropout_rate = args.userEncoder.dropout_rate
        
    def to(self, device):
        new_self = super(BaseOnlinePolicy, self).to(device)
        self.userEncoder.device = device
        self.userEncoder = self.userEncoder.to(device)
        return new_self
    
    def _define_params(self, args):
        self.userEncoder = BackboneUserEncoder(args, self.reader_stats, self.device)
        self.enc_dim = self.userEncoder.enc_dim
        self.state_dim = self.userEncoder.state_dim
        self.action_dim = self.slate_size
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction = 'none')

    def get_forward(self, feed_dict: dict):
        '''
        @input:
        - feed_dict: {
            'observation':{
                'user_profile':{
                    'user_id': (B,)
                    'uf_{feature_name}': (B,feature_dim), the user features}
                'user_history':{
                    'history': (B,max_H)
                    'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
            'candidates':{
                'item_id': (B,L) or (1,L), the target item
                'item_{feature_name}': (B,L,feature_dim) or (1,L,feature_dim), the target item features}
            'epsilon': scalar, 
            'do_explore': boolean,
            'candidates': {
                'item_id': (B,L) or (1,L), the target item
                'item_{feature_name}': (B,L,feature_dim) or (1,L,feature_dim), the target item features},
            'action_dim': slate size K,
            'action': (B,K),
            'response': {
                'reward': (B,),
                'immediate_response': (B,K*n_feedback)},
            'is_train': boolean
        }
        @output:
        - out_dict: {
            'state': (B,state_dim), 
            'prob': (B,K),
            'action': (B,K),
            'reg': scalar}
        '''
        observation = feed_dict['observation']
        # observation --> user state
        state_dict = self.get_user_state(observation)
        # user state + candidates --> dict(state, prob, action, reg)
        out_dict = self.generate_action(state_dict, feed_dict)
        out_dict['state'] = state_dict['state']
        out_dict['reg'] = state_dict['reg'] + out_dict['reg']
        return out_dict
    
    def get_user_state(self, observation):
        feed_dict = {}
        feed_dict.update(observation['user_profile'])
        feed_dict.update(observation['user_history'])
        B = feed_dict['user_id'].shape[0]
        return self.userEncoder(feed_dict, B)
    
    def get_loss_observation(self):
        return ['loss']
    
    def generate_action(self, state_dict, feed_dict):
        '''
        This function will be called in the following places:
        * OnlineAgent.run_episode_step() with {'action': None, 'response': None, 
                                               'epsilon': >0, 'do_explore': True, 'is_train': False}
        * OnlineAgent.step_train() with {'action': tensor, 'response': {'reward': , 'immediate_response': }, 
                                         'epsilon': 0, 'do_explore': False, 'is_train': True}
        * OnlineAgent.test() with {'action': None, 'response': None, 
                                   'epsilon': 0, 'do_explore': False, 'is_train': False}
        
        @input:
        - state_dict: {'state': (B, state_dim), ...}
        - feed_dict: {'candidates': ...}
        @output:
        - out_dict: {'prob': (B, K), 
                     'action': (B, K), 
                     'reg': scalar}
        '''
        pass