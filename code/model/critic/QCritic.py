import torch.nn.functional as F
import torch.nn as nn
import torch

from model.components import DNN
from utils import get_regularization

class QCritic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        '''
        parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[128], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--critic_dropout_rate', type=float, default=0.1, 
                            help='dropout rate in deep layers')
        return parser
    
    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim
        self.net = DNN(self.state_dim + self.action_dim, args.critic_hidden_dims, 1, 
                       dropout_rate = args.critic_dropout_rate, do_batch_norm = True)
        
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'state': (B, state_dim), 'action': (B, action_dim)}
        '''
        state_emb = feed_dict['state'].view(-1, self.state_dim)
        action_emb = feed_dict['action'].view(-1, self.action_dim)
        Q = self.net(torch.cat((state_emb, action_emb), dim = -1)).view(-1)
        reg = get_regularization(self.net)
        return {'q': Q, 'reg': reg}