# from model.simulator.KRUserRetention import KRUserRetention
from model.general import BaseModel
from model.components import DNN
import torch
import torch.nn as nn

class KRUserRetention(BaseModel):
    '''
    KuaiRand user retention model
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - sess_enc_dim
        - attn_n_head
        - transformer_d_forward
        - transformer_n_layer
        - output_hidden_dims
        - dropout_rate
        - from BaseModel:
            - model_path
            - loss
            - l2_coef
        '''
        parser = BaseModel.parse_model_args(parser)
        
        parser.add_argument('--enc_dim', type=int, default=32, 
                            help='session encoding size')
        parser.add_argument('--attn_n_head', type=int, default=4, 
                            help='number of attention heads in transformer')
        parser.add_argument('--transformer_d_forward', type=int, default=64, 
                            help='forward layer dimension in transformer')
        parser.add_argument('--transformer_n_layer', type=int, default=2, 
                            help='number of encoder layers in transformer')
        parser.add_argument('--output_hidden_dims', type=int, nargs='+', default=[128], 
                            help='hidden dimensions')
        parser.add_argument('--dropout_rate', type=float, default=0.1, 
                            help='dropout rate in deep layers')
        return parser
        
    def __init__(self, args, reader_stats, device):
        self.enc_dim = args.enc_dim
        self.attn_n_head = args.attn_n_head
        self.transformer_d_forward = args.transformer_d_forward
        self.transformer_n_layer = args.transformer_n_layer
        self.output_hidden_dims = args.output_hidden_dims
        self.dropout_rate = args.dropout_rate
        super().__init__(args, reader_stats, device)
        self.ce_loss = nn.CrossEntropyLoss(reduction = 'none')
        
    def to(self, device):
        new_self = super(KRUserRetention, self).to(device)
        new_self.attn_mask = new_self.attn_mask.to(device)
        new_self.pos_emb_getter = new_self.pos_emb_getter.to(device)
        return new_self

    def _define_params(self, args):
        stats = self.reader_stats
        
#         self.n_user = stats['n_user']
        self.sess_emb_dim = stats['enc_dim']
        self.state_dim = 2*args.enc_dim
        self.output_dim = stats['max_return_day']
        
        # user's average next day return probability
#         self.userReturnP = nn.Embedding(self.n_user+1, 1)
#         self.userReturnP.weight = torch.ones_like(self.userReturnP.weight) * 0.7
        
        # feedback embedding
        self.dayGapEmb = nn.Embedding(self.output_dim+1, args.enc_dim)
        
        # session embedding kernel encoder
        self.sessFeatureKernel = nn.Linear(self.sess_emb_dim, args.enc_dim)
        self.encDropout = nn.Dropout(self.dropout_rate)
        self.encNorm = nn.LayerNorm(args.enc_dim)
        
        # positional embedding
        self.max_len = stats['max_sess_len']
        self.posEmb = nn.Embedding(self.max_len, args.enc_dim)
        self.pos_emb_getter = torch.arange(self.max_len, dtype = torch.long)
        self.attn_mask = ~torch.tril(torch.ones((self.max_len,self.max_len), dtype=torch.bool))
        
        # sequence encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.state_dim, dim_feedforward = args.transformer_d_forward, 
                                                   nhead=args.attn_n_head, dropout = args.dropout_rate, 
                                                   batch_first = True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.transformer_n_layer)
        self.stateNorm = nn.LayerNorm(self.state_dim)
        
        # output DNN
        self.outputModule = DNN(self.state_dim, args.output_hidden_dims, self.output_dim, 
                                dropout_rate = args.dropout_rate, do_batch_norm = True)

    def get_forward(self, feed_dict: dict):
        '''
        @input:
        - feed_dict: {
            'user_id': (B,)
            'return_day': (B,)
            'history_encoding': (B,max_H,enc_dim)
            'history_response': (B,max_H)
            ... (irrelevant input)
        }
        @output:
        - out_dict: {'preds': (B,output_dim), 'state': (B,state_dim), 'reg': scalar}
        '''
        # (B,)
        users = feed_dict['user_id']
        B = users.shape[0]
        
        # (B, output_dim)
#         user_p = torch.clamp(self.userReturnP[users].view(B,1), 0, 1)
#         p_bias = [((1 - user_p) ** i) * user_p for i in range(self.output_dim - 1)] + [(1 - userp) ** (self.output_dim - 1)]
#         p_bias = torch.cat(p_bias, dim = 
        
        # user encoding
        state_encoder_output = self.encode_state(feed_dict, B)
        # (B, state_dim)
        user_state = state_encoder_output['state'].view(B,self.state_dim)

        # output
        # (B, output_dim)
        preds = self.outputModule(user_state)

        # regularization terms
        reg = self.get_regularization(self.sessFeatureKernel, self.posEmb, self.transformer, self.outputModule)
        reg = reg + state_encoder_output['reg']
        
        return {'preds': preds, 'state': user_state, 'reg': reg}
    
    def encode_state(self, feed_dict, B):
        '''
        @input:
        - feed_dict: {
            'user_id': (B,)
            'history_encoding': (B,max_H,enc_dim)
            'history_response': (B,max_H)
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
        history_enc = self.sessFeatureKernel(feed_dict['history_encoding']).view(B,self.max_len,self.enc_dim)
        # (1, max_H, enc_dim)
        pos_emb = self.posEmb(self.pos_emb_getter).view(1,self.max_len,self.enc_dim)
        # (B, max_H, enc_dim)
        seq_enc_feat = self.encNorm(self.encDropout(history_enc + pos_emb))
        # (B, max_H, enc_dim)
        history_feedback_emb = self.dayGapEmb(feed_dict['history_response']).view(B,self.max_len,self.enc_dim)
        # (B, max_H, state_dim)
        seq_enc = torch.cat((seq_enc_feat, history_feedback_emb), dim = -1)
        # (B, max_H, state_dim)
        output_seq = self.transformer(seq_enc, mask = self.attn_mask)
        # (B, state_dim)
        hist_enc = output_seq[:,-1,:].view(B,self.state_dim)
        state = self.stateNorm(hist_enc)
        return {'output_seq': output_seq, 'state': state, 'reg': torch.mean(history_feedback_emb*history_feedback_emb)}
    
    
    def get_loss(self, feed_dict: dict, out_dict: dict):
        """
        @input:
        - feed_dict: {...}
        - out_dict: {"preds":, "reg":}
        
        Loss terms implemented:
        - BCE
        """
        B = feed_dict['user_id'].shape[0]
        # (B, output_dim)
        preds = out_dict['preds'].view(B,self.output_dim)
        # (B,)
        targets = feed_dict['return_day']
        
        if self.loss_type == 'ce':
            loss = self.ce_loss(torch.softmax(preds, dim = -1), targets)
        else:
            raise NotImplemented
        out_dict['loss'] = torch.mean(loss) + self.l2_coef * out_dict['reg']
        return out_dict