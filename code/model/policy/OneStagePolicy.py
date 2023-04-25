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
            - state_user_latent_dim
            - state_item_latent_dim
            - state_transformer_enc_dim
            - state_transformer_n_head
            - state_transformer_d_forward
            - state_transformer_n_layer
            - state_dropout_rate
            - from BaseModel:
                - model_path
                - loss
                - l2_coef
        '''
        parser = BackboneUserEncoder.parse_model_args(parser)
        return parser
    
    def __init__(self, *input_args):
        args, env = input_args
        self.slate_size = env.slate_size
        super().__init__(args, env.reader.get_statistics(), args.device)
        self.display_name = "OneStagePolicy"
        self.dropout_rate = self.user_encoder.dropout_rate
        
    def to(self, device):
        new_self = super(OneStagePolicy, self).to(device)
        self.user_encoder.device = device
        self.user_encoder = self.user_encoder.to(device)
        return new_self
    
    def _define_params(self, args, reader_stats):
        self.user_encoder = BackboneUserEncoder(args, reader_stats)
        self.enc_dim = self.user_encoder.enc_dim
        self.state_dim = self.user_encoder.state_dim
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
        return self.user_encoder.get_forward(feed_dict)
    
    def get_loss_observation(self):
        return ['loss']
    
    def generate_action(self, state_dict, feed_dict):
        '''
        List generation provides three main types of exploration:
        * Greedy top-K: no exploration
        * Categorical sampling: probabilistic exploration
        * Uniform sampling: random exploration
        
        This function will be called in the following places:
        * agent.run_episode_step() during online inference, corresponding input_dict:
            {'action': None, 'response': None, 'epsilon': >0, 'do_explore': True, 'is_train': False}
        * agent.step_train() during online training, correpsonding input_dict:
            {'action': tensor, 'response': see buffer.sample@output - user_response, 
             'epsilon': 0, 'do_explore': False, 'is_train': True}
        * agent.test() during test, corresponding input_dict:
            {'action': None, 'response': None, 'epsilon': 0, 'do_explore': False, 'is_train': False}
        
        @input:
        - state_dict: {'state': (B, state_dim), ...}
        - feed_dict: same as self.get_forward@input - feed_dict
        @output:
        - out_dict: {'prob': (B, K), 
                     'action': (B, K), 
                     'indices': (B, K),
                     'reg': scalar}
        '''
        pass