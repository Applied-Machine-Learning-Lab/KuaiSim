import torch
import torch.nn as nn

from model.general import BaseModel
from model.policy.BackboneUserEncoder import BackboneUserEncoder

class BaseOnlinePolicy(BaseModel):
    '''
    Pointwise model
    '''
    
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
            - dropout_rate
            - from BaseModel:
                - model_path
                - loss
                - l2_coef
        '''
        parser = BackboneUserEncoder.parse_model_args(parser)
#         parser.add_argument('--score_clip', type=float, default=2.0, 
#                             help='ranking scores will be [-clip, +clip]')
        return parser
        
    def __init__(self, args, env, device):
        self.slate_size = args.slate_size # from environment arguments
        # BaseModel initialization: 
        # - reader_stats, model_path, loss_type, l2_coef, no_reg, device
        # - _define_params(args): enc_dim, state_dim, action_dim
#         self.score_clip = args.score_clip
        super().__init__(args, env.reader.get_statistics(), device)
        self.display_name = "BaseOnlinePolicy"
        
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
        
#         # user_state2condition layer
#         self.state2z = nn.Linear(self.state_dim, self.enc_dim)
#         self.zNorm = nn.LayerNorm(self.enc_dim)
        
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
#         candidates = feed_dict['candidates']
        # observation --> user state
        state_encoder_output = self.get_user_state(observation)
        # (B, state_dim)
        user_state = state_encoder_output['state']
        # user state --> prob, action
        out_dict = self.generate_action(user_state, feed_dict)

        out_dict['state'] = user_state
        out_dict['reg'] = state_encoder_output['reg'] + out_dict['reg']
        
        return out_dict
    
    def get_user_state(self, observation):
        feed_dict = {}
        feed_dict.update(observation['user_profile'])
        feed_dict.update(observation['user_history'])
        B = feed_dict['user_id'].shape[0]
        return self.userEncoder(feed_dict, B)
    
    def get_loss_observation(self):
        return ['loss']
    
    def generate_action(self, user_state, feed_dict):
        '''
        This function will be called in the following places:
        * OnlineAgent.run_episode_step() with {'action': None, 'response': None, 
                                               'epsilon': >0, 'do_explore': True, 'is_train': False}
        * OnlineAgent.step_train() with {'action': tensor, 'response': {'reward': , 'immediate_response': }, 
                                         'epsilon': 0, 'do_explore': False, 'is_train': True}
        * OnlineAgent.test() with {'action': None, 'response': None, 
                                   'epsilon': 0, 'do_explore': False, 'is_train': False}
        
        @input:
        - user_state
        - feed_dict
        @output:
        - out_dict: {'prob': (B, K), 
                     'action': (B, K), 
                     'reg': scalar}
        '''
        pass

    
    def get_loss(self, feed_dict, out_dict):
        '''
        @input:
        - feed_dict: same as get_forward@input-feed_dict
        - out_dict: {
            'state': (B,state_dim), 
            'prob': (B,K),
            'action': (B,K),
            'reg': scalar, 
            'immediate_response': (B,K),
            'reward': (B,)}
        @output
        - loss
        '''
        pass