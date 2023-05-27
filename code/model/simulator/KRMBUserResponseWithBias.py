from matplotlib.pyplot import axes, axis
import torch
import torch.nn as nn

from model.general import BaseModel
from model.components import DNN
from model.simulator.KRMBUserResponse import KRMBUserResponse


class KRMBUserResponseWithBias(KRMBUserResponse):
    '''
    KuaiRand Multi-Behavior user response model
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - user_latent_dim
        - item_latent_dim
        - enc_dim
        - attn_n_head
        - transformer_d_forward
        - transformer_n_layer
        - scorer_hidden_dims
        - dropout_rate
        - from BaseModel:
            - model_path
            - loss
            - l2_coef
        '''
        parser = KRMBUserResponse.parse_model_args(parser)
        return parser
        
    def log(self):
        print("KRMBUserResponseWithBias params:")
        super().log()
            
    def __init__(self, args, reader_stats, device):
        super().__init__(args, reader_stats, device)
        
    def to(self, device):
        new_self = super(KRMBUserResponseWithBias, self).to(device)
        return new_self

    def _define_params(self, args, reader_stats):
        super()._define_params(args, reader_stats)
        self.scorer = DNN(3*args.enc_dim, args.state_hidden_dims, self.feedback_dim * (args.enc_dim + 1), 
                          dropout_rate = args.dropout_rate, do_batch_norm = True)
    
        
    def get_pointwise_scores(self, user_state, item_enc, B):
        '''
        Get user-item pointwise interaction scores
        @input:
        - user_state: (B, state_dim)
        - item_enc: (B, -1, 1, enc_dim) for batch-wise candidates or (1, -1, 1, enc_dim) for universal candidates
        - B: batch size
        @output:
        - behavior_scores: (B, -1, n_feedback)
        '''
        # scoring
        # (B, 1, n_feedback, enc_dim+1)
        scorer_output = self.scorer(user_state).view(B,1,self.feedback_dim,self.enc_dim+1)
        # (B, 1, n_feedback, enc_dim)
        behavior_attn = self.stateNorm(scorer_output[:,:,:,:-1])
        # (B, 1, n_feedback)
        behavior_bias = scorer_output[:,:,:,-1]
        # (B, -1, n_feedback)
        point_scores = (behavior_attn * item_enc).mean(dim = -1).view(B,-1,self.feedback_dim)
        point_scores = point_scores + behavior_bias
        return point_scores, torch.mean(point_scores, dim = -1)
    
    