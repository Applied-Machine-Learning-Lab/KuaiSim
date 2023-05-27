from matplotlib.pyplot import axes, axis
import torch
import torch.nn as nn

from model.general import BaseModel
from model.components import DNN
import utils

class ActionTransformer(BaseModel):
    '''
    KuaiRand Multi-Behavior user response model
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - policy_user_latent_dim
        - policy_item_latent_dim
        - policy_enc_dim
        - policy_attn_n_head
        - policy_transformer_d_forward
        - policy_transformer_n_layer
        - policy_hidden_dims
        - policy_dropout_rate
        - from BaseModel:
            - model_path
            - loss
            - l2_coef
        '''
        parser = BaseModel.parse_model_args(parser)
        
        parser.add_argument('--policy_user_latent_dim', type=int, default=16, 
                            help='user latent embedding size')
        parser.add_argument('--policy_item_latent_dim', type=int, default=16, 
                            help='item latent embedding size')
        parser.add_argument('--policy_enc_dim', type=int, default=32, 
                            help='item encoding size')
        parser.add_argument('--policy_attn_n_head', type=int, default=4, 
                            help='number of attention heads in transformer')
        parser.add_argument('--policy_transformer_d_forward', type=int, default=64, 
                            help='forward layer dimension in transformer')
        parser.add_argument('--policy_transformer_n_layer', type=int, default=2, 
                            help='number of encoder layers in transformer')
        parser.add_argument('--policy_hidden_dims', type=int, nargs='+', default=[128], 
                            help='hidden dimensions')
        parser.add_argument('--policy_dropout_rate', type=float, default=0.1, 
                            help='dropout rate in deep layers')
        return parser
        
    def __init__(self, *input_args):
        args, env = input_args
        self.user_latent_dim = args.policy_user_latent_dim
        self.item_latent_dim = args.policy_item_latent_dim
        self.enc_dim = args.policy_enc_dim
        self.state_dim = 3*args.policy_enc_dim
        self.attn_n_head = args.policy_attn_n_head
        self.action_hidden_dims = args.policy_hidden_dims
        self.dropout_rate = args.policy_dropout_rate
        super().__init__(args, env.reader.get_statistics(), args.device)
        self.display_name = "ActionTransformer"
        self.bce_loss = nn.BCEWithLogitsLoss(reduction = 'none')
        
    def to(self, device):
        new_self = super(ActionTransformer, self).to(device)
        new_self.attn_mask = new_self.attn_mask.to(device)
        new_self.pos_emb_getter = new_self.pos_emb_getter.to(device)
        return new_self

    def _define_params(self, args, reader_stats):
        stats = reader_stats

        self.user_feature_dims = stats['user_feature_dims'] # {feature_name: dim}
        self.item_feature_dims = stats['item_feature_dims'] # {feature_name: dim}

        # user embedding
        self.uIDEmb = nn.Embedding(stats['n_user']+1, args.policy_user_latent_dim)
        self.uFeatureEmb = {}
        for f,dim in self.user_feature_dims.items():
            embedding_module = nn.Linear(dim, args.policy_user_latent_dim)
            self.add_module(f'UFEmb_{f}', embedding_module)
            self.uFeatureEmb[f] = embedding_module
            
        # item embedding
        self.iIDEmb = nn.Embedding(stats['n_item']+1, args.policy_item_latent_dim)
        self.iFeatureEmb = {}
        for f,dim in self.item_feature_dims.items():
            embedding_module = nn.Linear(dim, args.policy_item_latent_dim)
            self.add_module(f'IFEmb_{f}', embedding_module)
            self.iFeatureEmb[f] = embedding_module
        
        # feedback embedding
        self.feedback_types = stats['feedback_type']
        self.feedback_dim = stats['feedback_size']
        self.action_dim = self.feedback_dim
        self.xtr_dim = 2*self.feedback_dim
        self.feedbackEncoder = nn.Linear(self.feedback_dim, args.policy_enc_dim)
        
        # item embedding kernel encoder
        self.itemEmbNorm = nn.LayerNorm(args.policy_item_latent_dim)
        self.userEmbNorm = nn.LayerNorm(args.policy_user_latent_dim)
        self.itemFeatureKernel = nn.Linear(args.policy_item_latent_dim, args.policy_enc_dim)
        self.userFeatureKernel = nn.Linear(args.policy_user_latent_dim, args.policy_enc_dim)
        self.encDropout = nn.Dropout(self.dropout_rate)
        self.encNorm = nn.LayerNorm(args.policy_enc_dim)
        
        # positional embedding
        self.max_len = stats['max_seq_len']
        self.posEmb = nn.Embedding(self.max_len, args.policy_enc_dim)
        self.pos_emb_getter = torch.arange(self.max_len, dtype = torch.long)
        self.attn_mask = ~torch.tril(torch.ones((self.max_len,self.max_len), dtype=torch.bool))
        
        # sequence encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=2*args.policy_enc_dim, dim_feedforward = args.policy_transformer_d_forward, 
                                                   nhead=args.policy_attn_n_head, dropout = args.policy_dropout_rate, 
                                                   batch_first = True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.policy_transformer_n_layer)
        
        # DNN state encoder
        self.stateNorm = nn.LayerNorm(args.policy_enc_dim)
        
        # DNN action layer
        self.actionModule = DNN(self.state_dim, args.policy_hidden_dims, self.feedback_dim,
                                dropout_rate = args.policy_dropout_rate, do_batch_norm = True)
        #self.actionModule = torch.nn.Sigmoid(self.actionModule)

    def forward(self, feed_dict: dict, return_prob = True) -> dict:
        out_dict = self.get_forward(feed_dict)
        out_dict["preds"] = out_dict["action"]
        if return_prob:
            out_dict["probs"] = out_dict["action"]
        return out_dict
    
    def get_forward(self, feed_dict: dict):
        '''
        @input:
        - feed_dict: {'user_profile':{
                           'user_id': (B,)
                           'uf_{feature_name}': (B,feature_dim), the user features}
                      'user_history':{
                           'history': (B,max_H)
                           'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
            ... (irrelevant input)
        }
        @output:
        - out_dict: {'preds': (B,-1,n_feedback), 'reg': scalar}
        '''
        
        batch = {}
        batch.update(feed_dict['user_profile'])
        batch.update(feed_dict['user_history'])
        batch = utils.wrap_batch(batch, self.device)
        
        B = batch['user_id'].shape[0]
        
        # user encoding
        state_encoder_output = self.encode_state(batch, B)
        # (B, state_dim)
        user_state = state_encoder_output['state'].view(B,self.state_dim)

        # get action
        # (B, n_feedback)
        action_emb = self.actionModule(user_state)
        # regularization terms
        reg = self.get_regularization(self.feedbackEncoder, 
                                      self.itemFeatureKernel, self.userFeatureKernel, 
                                      self.posEmb, self.transformer, self.actionModule)
        reg = reg + state_encoder_output['reg']
        
        return {'action': action_emb, 'state': user_state, 'reg': reg}
    
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
        # user history
        # (B, max_H, enc_dim)
        history_enc, history_reg = self.get_item_encoding(feed_dict['history'], 
                                             {f:feed_dict[f'history_if_{f}'] for f in self.iFeatureEmb}, B)
        history_enc = history_enc.view(B, self.max_len, self.enc_dim)
        # (1, max_H, enc_dim)
        pos_emb = self.posEmb(self.pos_emb_getter).view(1,self.max_len,self.enc_dim)
        # (B, max_H, enc_dim)
        seq_enc_feat = self.encNorm(self.encDropout(history_enc + pos_emb))
        # (B, max_H, enc_dim)
        feedback_emb = self.get_response_embedding(feed_dict, B)
        # (B, max_H, 2*enc_dim)
        seq_enc = torch.cat((seq_enc_feat, feedback_emb), dim = -1)
        # (B, max_H, 2*enc_dim)
        output_seq = self.transformer(seq_enc, mask = self.attn_mask)
        # (B, 2*enc_dim)
        hist_enc = output_seq[:,-1,:].view(B,2*self.enc_dim)
        # user features
        # (B, enc_dim), scalar
        user_enc, user_reg = self.get_user_encoding(feed_dict['user_id'], 
                                          {k[3:]:v for k,v in feed_dict.items() if k[:3] == 'uf_'}, B)
        # (B, enc_dim)
        user_enc = self.encNorm(self.encDropout(user_enc)).view(B,self.enc_dim)
        # (B, 3*enc_dim)
        state = torch.cat([hist_enc,user_enc], 1)
        return {'output_seq': output_seq, 'state': state, 'reg': user_reg + history_reg}
    
    def get_user_encoding(self, user_ids, user_features, B):
        '''
        @input:
        - user_ids: (B,)
        - user_features: {'uf_{feature_name}': (B, feature_dim)}
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
        - item_ids: (B,) or (B,H)
        - item_features: {'if_{feature_name}': (B,feature_dim) or (B,H,feature_dim)}
        '''
        # (B, 1, i_latent_dim) or (B, H, i_latent_dim)
        item_id_emb = self.iIDEmb(item_ids).view(B,-1,self.item_latent_dim)
        L = item_id_emb.shape[1]
        # [(B, 1, i_latent_dim)] * n_item_feature or [(B, H, i_latent_dim)] * n_item_feature
        item_feature_emb = [item_id_emb]
        for f,fEmbModule in self.iFeatureEmb.items():
            f_dim = self.item_feature_dims[f]
            item_feature_emb.append(fEmbModule(item_features[f].view(B,L,f_dim)).view(B,-1,self.item_latent_dim))
        # (B, 1, n_item_feature+1, i_latent_dim) or (B, H, n_item_feature+1, i_latent_dim)
        combined_item_emb = torch.cat(item_feature_emb, -1).view(B, L, -1, self.item_latent_dim)
        combined_item_emb = self.itemEmbNorm(combined_item_emb)
        # (B, 1, enc_dim) or (B, H, enc_dim)
        encoding = self.itemFeatureKernel(combined_item_emb).sum(2)
        encoding = encoding.view(B, -1, self.enc_dim)
        encoding = self.encNorm(encoding)
        # regularization
        reg = torch.mean(item_id_emb * item_id_emb)
        return encoding, reg
        
    def get_response_embedding(self, feed_dict, B):
        resp_list = []
        for f in self.feedback_types:
            # (B, max_H)
            resp = feed_dict[f'history_{f}'].view(B, self.max_len)
            resp_list.append(resp)
        # (B, max_H, n_feedback)
        combined_resp = torch.cat(resp_list, -1).view(B,self.max_len,self.feedback_dim)
        # (B, max_H, i_latent_dim)
        resp_emb = self.feedbackEncoder(combined_resp)
        return resp_emb
    
    def get_loss(self, feed_dict: dict, out_dict: dict):
        pass
    
    