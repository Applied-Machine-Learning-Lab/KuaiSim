import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from model.components import DNN
from utils import get_regularization
from model.score_func import dot_scorer
    
class SASRec(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        
        '''
        parser.add_argument('--sasrec_n_layer', type=int, default=2, 
                            help='number of transformer layers')
        parser.add_argument('--sasrec_d_model', type=int, default=32, 
                            help='input embedding size of transformer')
        parser.add_argument('--sasrec_d_forward', type=int, default=64, 
                            help='input embedding size of transformer')
        parser.add_argument('--sasrec_n_head', type=int, default=4, 
                            help='number of attention heads in transformer layers')
        parser.add_argument('--sasrec_dropout', type=float, default=0.1, 
                            help='number of attention heads in transformer layers')
        return parser
    
    def __init__(self, args, environment):
        '''
        action_space = {'item_id': ('nominal', stats['n_item']), 
                        'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive']), 
                            'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}
        '''
        super().__init__()
        self.n_layer = args.sasrec_n_layer
        self.d_model = args.sasrec_d_model
        self.n_head = args.sasrec_n_head
        self.dropout_rate = args.sasrec_dropout
        # item space
        self.item_space = environment.action_space['item_id'][1]
        self.item_dim = environment.action_space['item_feature'][1]
        self.maxlen = environment.observation_space['history'][1]
        self.state_dim = self.d_model
        self.action_dim = self.d_model
        # policy network modules
        self.item_map = nn.Linear(self.item_dim, self.d_model)
        self.pos_emb = nn.Embedding(self.maxlen, self.d_model)
        self.pos_emb_getter = torch.arange(self.maxlen, dtype = torch.long)
        self.emb_dropout = nn.Dropout(self.dropout_rate)
        self.emb_norm = nn.LayerNorm(self.d_model)
        self.attn_mask = ~torch.tril(torch.ones((self.maxlen,self.maxlen), dtype=torch.bool))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, dim_feedforward = args.sasrec_d_forward, 
                                                   nhead=self.n_head, dropout = self.dropout_rate, batch_first = True)
#         self.layernorm = nn.LayerNorm(self.d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layer)
        
        
    def score(self, action_emb, item_emb, do_softmax = True):
        '''
        @input:
        - action_emb: (B, (i_dim+1))
        - item_emb: (B, L, i_dim) or (1, L, i_dim)
        @output:
        - scores: (B, L)
        '''
        item_emb = self.item_map(item_emb)
        output = dot_scorer(action_emb, item_emb, self.d_model)
        if do_softmax:
            return torch.softmax(output, dim = -1)
        else:
            return output
        
    def get_scorer_parameters(self):
        return self.item_map.parameters()
    
    def encode_state(self, feed_dict):
        user_history = feed_dict['history_features'] 
#         history_mask = feed_dict['history_mask']
        # (1, H, d_model)
        pos_emb = self.pos_emb(self.pos_emb_getter.to(user_history.device)).view(1,self.maxlen,self.d_model)
        # (B, H, d_model)
        hist_item_emb = self.item_map(user_history).view(-1,self.maxlen,self.d_model)
        hist_item_emb = self.emb_norm(self.emb_dropout(hist_item_emb + pos_emb))
        
#         hist_item_emb = hist_item_emb * history_mask.view(-1,self.maxlen,1)
        # (B, H, d_model)
        output_seq = self.transformer(hist_item_emb, mask = self.attn_mask.to(user_history.device))
        return {'output_seq': output_seq, 'state_emb': output_seq[:,-1,:]}

    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'user_profile': (B, user_dim), 
                    'history_features': (B, H, item_dim),
                    'history_mask': (B,),
                    'candidate_features': (B, L, item_dim) or (1, L, item_dim)}
        @model:
        - user_profile --> user_emb (B,1,f_dim)
        - history_items --> history_item_emb (B,H,f_dim)
        - (Q:user_emb, K&V:history_item_emb) --(multi-head attn)--> user_state (B,1,f_dim)
        - user_state --> action_prob (B,n_item)
        @output:
        - out_dict: {"action_emb": (B,action_dim), 
                     "state_emb": (B,f_dim),
                     "reg": scalar,
                     "action_prob": (B,L), include probability score when candidate_features are given}
        '''
        hist_enc = self.encode_state(feed_dict)
        # user embedding (B,1,d_model)
        user_state = hist_enc['state_emb'].view(-1,self.d_model)
        B = user_state.shape[0]
        # action embedding (B, d_model * )
        action_emb = user_state
#         print(action_emb[0])
#         print(action_emb.shape)
#         item_map_action = torch.cat((self.item_map.weight.view(1,-1), self.item_map.bias.view(1,-1)), dim = 1)
#         action_emb = torch.cat((user_state, torch.tile(item_map_action, (B,1))), dim = 1).view(B,-1)
        # regularization terms
        reg = get_regularization(self.item_map, self.transformer)
        # output
        out_dict = {'action_emb': action_emb, 
                    'state_emb': user_state,
                    'seq_emb': hist_enc['output_seq'],
                    'reg': reg}
#         if 'candidate_features' in feed_dict:
#             # action prob (B,L)
#             action_prob = self.score(action_emb, feed_dict['candidate_features'], feed_dict['do_softmax'])
#             out_dict['action_prob'] = action_prob
#             out_dict['candidate_ids'] = feed_dict['candidate_ids']
        return out_dict