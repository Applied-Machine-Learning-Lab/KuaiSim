from matplotlib.pyplot import axes, axis
import torch
import torch.nn as nn

from model.general import BaseModel
from model.components import DNN
from model.simulator.KRMBUserResponse import KRMBUserResponse

class KRMBUserResponse_SubFB(KRMBUserResponse):
    '''
    KuaiRand Multi-Behavior user response model with subset of feedback types
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - feedback_subset
        - from KRMBUserResponse:
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
        parser.add_argument('--feedback_subset', type=int, nargs='+', default=[0,1,2], 
                            help='indices of selected feedback types')
        return parser
        
    def log(self):
        print("KRMBUserResponse_SubFB params:")
        super().log()
            
    def __init__(self, args, reader_stats, device):
        self.feedback_subset = args.feedback_subset
        super().__init__(args, reader_stats, device)

    def _define_params(self, args, reader_stats):
        reader_stats['feedback_type'] = [reader_stats['feedback_type'][i] for i in args.feedback_subset]
        reader_stats['feedback_size'] = len(reader_stats['feedback_type'])
        super()._define_params(args, reader_stats)
    
    