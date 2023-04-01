from matplotlib.pyplot import axes, axis
import torch
import torch.nn as nn

from model.general import BaseModel
from model.components import DNN

class BackboneUserEncoder(BaseModel):
    '''
    KuaiRand Multi-Behavior user response model
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
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
        parser = BaseModel.parse_model_args(parser)
        
        parser.add_argument('--user_latent_dim', type=int, default=16, 
                            help='user latent embedding size')
        parser.add_argument('--item_latent_dim', type=int, default=16, 
                            help='item latent embedding size')
        parser.add_argument('--transformer_enc_dim', type=int, default=32, 
                            help='item encoding size')
        parser.add_argument('--transformer_n_head', type=int, default=4, 
                            help='number of attention heads in transformer')
        parser.add_argument('--transformer_d_forward', type=int, default=64, 
                            help='forward layer dimension in transformer')
        parser.add_argument('--transformer_n_layer', type=int, default=2, 
                            help='number of encoder layers in transformer')
        parser.add_argument('--state_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of final state encoding layers')
        parser.add_argument('--state_dropout_rate', type=float, default=0.1, 
                            help='dropout rate in deep layers of state encoder')
        return parser
        
    def __init__(self, args, reader_stats, device):
        self.user_latent_dim = args.user_latent_dim
        self.item_latent_dim = args.item_latent_dim
        self.enc_dim = args.transformer_enc_dim
        self.state_dim = self.enc_dim
        self.attn_n_head = args.transformer_n_head
        self.state_hidden_dims = args.state_hidden_dims
        self.dropout_rate = args.state_dropout_rate
        super().__init__(args, reader_stats, device)
        
    def to(self, device):
        new_self = super(BackboneUserEncoder, self).to(device)
        new_self.attn_mask = new_self.attn_mask.to(device)
        new_self.pos_emb_getter = new_self.pos_emb_getter.to(device)
        return new_self

    def _define_params(self, args):
        stats = self.reader_stats

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
        self.feedbackEncoder = nn.Linear(self.feedback_dim, args.transformer_enc_dim)
        
        # item embedding kernel encoder
        self.itemEmbNorm = nn.LayerNorm(args.item_latent_dim)
        self.userEmbNorm = nn.LayerNorm(args.user_latent_dim)
        self.itemFeatureKernel = nn.Linear(args.item_latent_dim, args.transformer_enc_dim)
        self.userFeatureKernel = nn.Linear(args.user_latent_dim, args.transformer_enc_dim)
        self.encDropout = nn.Dropout(self.dropout_rate)
        self.encNorm = nn.LayerNorm(args.transformer_enc_dim)
        
        # positional embedding
        self.max_len = stats['max_seq_len']
        self.posEmb = nn.Embedding(self.max_len, args.transformer_enc_dim)
        self.pos_emb_getter = torch.arange(self.max_len, dtype = torch.long)
        self.attn_mask = ~torch.tril(torch.ones((self.max_len,self.max_len), dtype=torch.bool))
        
        # sequence encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=2*args.transformer_enc_dim, 
                                                   dim_feedforward = args.transformer_d_forward, 
                                                   nhead=args.transformer_n_head, dropout = args.dropout_rate, 
                                                   batch_first = True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.transformer_n_layer)
        
        # DNN state encoder
        self.stateNorm = nn.LayerNorm(self.state_dim)
        self.finalStateLayer = DNN(3*args.transformer_enc_dim, args.state_hidden_dims, self.state_dim,
                                dropout_rate = args.dropout_rate, do_batch_norm = True)
        
        #self.actionModule = torch.nn.Sigmoid(self.actionModule)

    def get_forward(self, feed_dict: dict):
        '''
        @input:
        - feed_dict: {
            'user_id': (B,)
            'uf_{feature_name}': (B,feature_dim), the user features
            'item_id': (B,L), the target item
            'if_{feature_name}': (B,L,feature_dim), the target item features
            'history': (B,max_H)
            'history_if_{feature_name}': (B,max_H,feature_dim), the history item features
        }
        @output:
        - out_dict: {'state': (B, state_dim), 
                    'reg': scalar}
        '''
        B = feed_dict['user_id'].shape[0]
        # user encoding
        state_encoder_output = self.encode_state(feed_dict, B)
        # regularization terms
        reg = self.get_regularization(self.feedbackEncoder, 
                                      self.itemFeatureKernel, self.userFeatureKernel, 
                                      self.posEmb, self.transformer)
        reg = reg + state_encoder_output['reg']
        
        return {'state': state_encoder_output['state'], 
                'reg': reg}
    
    def encode_state(self, feed_dict, B):
        '''
        @input:
        - feed_dict: {
            'user_id': (B,)
            'uf_{feature_name}': (B,feature_dim), the user features
            'history': (B,max_H)
            'history_if_{feature_name}': (B,max_H,feature_dim), the history item features
            ... (irrelevant input)
        }
        - B: batch size
        @output:
        - out_dict:{
            'out_seq': (B,max_H,2*enc_dim)
            'state': (B,n_feedback*enc_dim)
            'reg': scalar
        }
        '''
        # user history item encodings (B, max_H, enc_dim)
        history_enc, history_reg = self.get_item_encoding(feed_dict['history'], 
                                             {f:feed_dict[f'history_if_{f}'] for f in self.iFeatureEmb}, B)
        history_enc = history_enc.view(B, self.max_len, self.enc_dim)
        
        # positional encoding (1, max_H, enc_dim)
        pos_emb = self.posEmb(self.pos_emb_getter).view(1,self.max_len,self.enc_dim)
        
        # feedback embedding (B, max_H, enc_dim)
        feedback_emb = self.get_response_embedding({f: feed_dict[f'history_{f}'] for f in self.feedback_types}, B)
        
        # sequence item encoding (B, max_H, enc_dim)
        seq_enc_feat = self.encNorm(self.encDropout(history_enc + pos_emb))
        # (B, max_H, 2*enc_dim)
        seq_enc = torch.cat((seq_enc_feat, feedback_emb), dim = -1)
        
        # transformer output (B, max_H, 2*enc_dim)
        output_seq = self.transformer(seq_enc, mask = self.attn_mask)
        
        # user history encoding (B, 2*enc_dim)
        hist_enc = output_seq[:,-1,:].view(B,2*self.enc_dim)
        
        # static user profile features
        # (B, enc_dim), scalar
        user_enc, user_reg = self.get_user_encoding(feed_dict['user_id'], 
                                          {k[3:]:v for k,v in feed_dict.items() if k[:3] == 'uf_'}, B)
        # (B, enc_dim)
        user_enc = self.encNorm(self.encDropout(user_enc)).view(B,self.enc_dim)
        
        # user state (B, 3*enc_dim) combines user history and user profile features
        state = torch.cat([hist_enc,user_enc], 1)
        # (B, enc_dim)
        state = self.stateNorm(self.finalStateLayer(state))
        return {'output_seq': output_seq, 'state': state, 'reg': user_reg + history_reg}
    
    def get_user_encoding(self, user_ids, user_features, B):
        '''
        @input:
        - user_ids: (B,)
        - user_features: {'uf_{feature_name}': (B, feature_dim)}
        @output:
        - encoding: (B, enc_dim)
        - reg: scalar
        '''
        # (B, 1, u_latent_dim)
        user_id_emb = self.uIDEmb(user_ids).view(B,1,self.user_latent_dim)
        # [(B, 1, u_latent_dim)] * n_user_feature
        user_feature_emb = [user_id_emb]
        for f,fEmbModule in self.uFeatureEmb.items():
            user_feature_emb.append(fEmbModule(user_features[f]).view(B,1,self.user_latent_dim))
        # (B, n_user_feature+1, u_latent_dim)
        combined_user_emb = torch.cat(user_feature_emb, 1)
        combined_user_emb = self.userEmbNorm(combined_user_emb)
        # (B, enc_dim)
        encoding = self.userFeatureKernel(combined_user_emb).sum(1)
        # regularization
        reg = torch.mean(user_id_emb * user_id_emb)
        return encoding, reg
        
    def get_item_encoding(self, item_ids, item_features, B):
        '''
        @input:
        - item_ids: (B,) or (B,L)
        - item_features: {'{feature_name}': (B,feature_dim) or (B,L,feature_dim)}
        @output:
        - encoding: (B, 1, enc_dim) or (B, L, enc_dim)
        - reg: scalar
        '''
        # (B, 1, i_latent_dim) or (B, L, i_latent_dim)
        item_id_emb = self.iIDEmb(item_ids).view(B,-1,self.item_latent_dim)
        L = item_id_emb.shape[1]
        # [(B, 1, i_latent_dim)] * n_item_feature or [(B, L, i_latent_dim)] * n_item_feature
        item_feature_emb = [item_id_emb]
        for f,fEmbModule in self.iFeatureEmb.items():
            f_dim = self.item_feature_dims[f]
            item_feature_emb.append(fEmbModule(item_features[f].view(B,L,f_dim)).view(B,-1,self.item_latent_dim))
        # (B, 1, n_item_feature+1, i_latent_dim) or (B, L, n_item_feature+1, i_latent_dim)
        combined_item_emb = torch.cat(item_feature_emb, -1).view(B, L, -1, self.item_latent_dim)
        combined_item_emb = self.itemEmbNorm(combined_item_emb)
        # (B, 1, enc_dim) or (B, L, enc_dim)
        encoding = self.itemFeatureKernel(combined_item_emb).sum(2)
        encoding = self.encNorm(encoding.view(B, -1, self.enc_dim))
        # regularization
        reg = torch.mean(item_id_emb * item_id_emb)
        return encoding, reg
        
    def get_response_embedding(self, resp_dict, B):
        '''
        @input:
        - resp_dict: {'{response}': (B, max_H)}
        @output:
        - resp_emb: (B, max_H, enc_dim)
        '''
        resp_list = []
        for f in self.feedback_types:
            # (B, max_H)
            resp = resp_dict[f].view(B, self.max_len)
            resp_list.append(resp)
        # (B, max_H, n_feedback)
        combined_resp = torch.cat(resp_list, -1).view(B,self.max_len,self.feedback_dim)
        # (B, max_H, enc_dim)
        resp_emb = self.feedbackEncoder(combined_resp)
        return resp_emb
    