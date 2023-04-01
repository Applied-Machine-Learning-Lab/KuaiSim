from matplotlib.pyplot import axes, axis
import torch
import torch.nn as nn

from model.simulator.KRMBUserResponse import KRMBUserResponse
from model.components import DNN

class KRMBUserResponse_MaxOut(KRMBUserResponse):
    '''
    KuaiRand Multi-Behavior user response model with ensemble-maxout scorer:
    - For each response, multiple output heads give proposals, and the scorer chooses the max one as output
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - n_ensemble
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
        parser.add_argument('--n_ensemble', type=int, default=2, 
                            help='item encoding size')
        return parser
        
    def log(self):
        print("KRMBUserResponse_MaxOut params:")
        print(f"\tn_ensemble: {self.n_ensemble}")
        super().log()
            
    def __init__(self, args, reader_stats, device):
        self.n_ensemble = args.n_ensemble
        super().__init__(args, reader_stats, device)
        
    def to(self, device):
        return super(KRMBUserResponse_MaxOut, self).to(device)
        

    def _define_params(self, args, reader_stats):
        stats = reader_stats

        self.user_feature_dims = stats['user_feature_dims'] # {feature_name: dim}
        self.item_feature_dims = stats['item_feature_dims'] # {feature_name: dim}

        # user embedding
        self.uIDEmb = nn.Embedding(stats['n_user']+1, args.user_latent_dim)
        self.uFeatureEmb = {}
        for f,dim in self.user_feature_dims.items():
            embedding_module = nn.Linear(dim, args.user_latent_dim)
            self.add_module(f'UFEmb_{f}', embedding_module)
            self.uFeatureEmb[f] = embedding_module
            
        # item embedding
        self.iIDEmb = nn.Embedding(stats['n_item']+1, args.item_latent_dim)
        self.iFeatureEmb = {}
        for f,dim in self.item_feature_dims.items():
            embedding_module = nn.Linear(dim, args.item_latent_dim)
            self.add_module(f'IFEmb_{f}', embedding_module)
            self.iFeatureEmb[f] = embedding_module
        
        # feedback embedding
        self.feedback_types = stats['feedback_type']
        self.feedback_dim = stats['feedback_size']
        self.feedbackEncoder = nn.Linear(self.feedback_dim, args.enc_dim)
        self.set_behavior_hyper_weight(torch.ones(self.feedback_dim))
        
        # item embedding kernel encoder
        self.itemEmbNorm = nn.LayerNorm(args.item_latent_dim)
        self.userEmbNorm = nn.LayerNorm(args.user_latent_dim)
        self.itemFeatureKernel = nn.Linear(args.item_latent_dim, args.enc_dim)
        self.userFeatureKernel = nn.Linear(args.user_latent_dim, args.enc_dim)
        self.encDropout = nn.Dropout(self.dropout_rate)
        self.encNorm = nn.LayerNorm(args.enc_dim)
        
        # positional embedding
        self.max_len = stats['max_seq_len']
        self.posEmb = nn.Embedding(self.max_len, args.enc_dim)
        self.pos_emb_getter = torch.arange(self.max_len, dtype = torch.long)
        self.attn_mask = ~torch.tril(torch.ones((self.max_len,self.max_len), dtype=torch.bool))
        
        # sequence encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=2*args.enc_dim, dim_feedforward = args.transformer_d_forward, 
                                                   nhead=args.attn_n_head, dropout = args.dropout_rate, 
                                                   batch_first = True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.transformer_n_layer)
        self.state_dim = 3*args.enc_dim
        
        # DNN state encoder
        self.stateNorm = nn.LayerNorm(args.enc_dim)
        
        # DNN scorer
        self.output_dim = self.feedback_dim * args.n_ensemble
        self.scorer_hidden_dims = args.scorer_hidden_dims
        self.scorer = DNN(self.state_dim, args.state_hidden_dims, self.output_dim * args.enc_dim, 
                          dropout_rate = args.dropout_rate, do_batch_norm = True)

    def get_forward(self, feed_dict: dict):
        '''
        @input:
        - feed_dict: {
            'user_id': (B,)
            'uf_{feature_name}': (B,feature_dim), the user features
            'item_id': (B,), the target item
            'if_{feature_name}': (B,feature_dim), the target item features
            'history': (B,max_H)
            'history_if_{feature_name}': (B,max_H,feature_dim), the history item features
            ... (irrelevant input)
        }
        @output:
        - out_dict: {'preds': (B,-1,n_feedback), 'reg': scalar}
        '''
        B = feed_dict['user_id'].shape[0]
        
        # target item
        # (B, -1, 1, enc_dim), scalar
        item_enc, item_reg = self.get_item_encoding(feed_dict['item_id'], 
                                          {k[3:]:v for k,v in feed_dict.items() if k[:3] == 'if_'}, B)
        item_enc = item_enc.view(B,-1,1,self.enc_dim)
        
        # user encoding
        state_encoder_output = self.encode_state(feed_dict, B)
        # (B, 1, 3*enc_dim)
        user_state = state_encoder_output['state'].view(B,1,self.state_dim)

        # (B, -1, n_feedback), (B, -1, output_dim)
        behavior_scores, point_scores = self.get_pointwise_scores(user_state, item_enc, B)

        # regularization terms
        reg = self.get_regularization(self.feedbackEncoder, 
                                      self.itemFeatureKernel, self.userFeatureKernel, 
                                      self.posEmb, self.transformer, self.scorer)
        reg = reg + state_encoder_output['reg'] + item_reg
        return {'preds': behavior_scores, 'state': user_state, 'reg': reg}
    
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
        # (B, 1, n_feedback*n_ensemble, enc_dim)
        behavior_attn = self.scorer(user_state).view(B,1,self.output_dim,self.enc_dim)
        # (B, 1, n_feedback*n_ensemble, enc_dim)
        behavior_attn = self.stateNorm(behavior_attn)
        # (B, -1, n_feedback, n_ensemble)
        point_scores = (behavior_attn * item_enc).mean(dim = -1).view(B,-1,self.feedback_dim,self.n_ensemble)
        # (B, -1, n_feedback)
        behavior_scores, max_indices = torch.max(point_scores, dim = -1)
        return behavior_scores, point_scores