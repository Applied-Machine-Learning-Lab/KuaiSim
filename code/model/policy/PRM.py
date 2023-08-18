import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from model.components import DNN
from model.policy.TwoStageOnlinePolicy import TwoStageOnlinePolicy

class PRM(TwoStageOnlinePolicy):
    '''
    Pointwise model
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - from TwoStageOnlinePolicy:
            - initial_list_size
            - stage1_n_neg
            - stage1_state2z_hidden_dims
            - stage1_pos_offset
            - stage1_neg_offset
            - initial_loss_coef
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
        parser = TwoStageOnlinePolicy.parse_model_args(parser)
        parser.add_argument('--prm_pv_input_dim', type=int, default=32, 
                            help='input size of PV module of PRM')
        parser.add_argument('--prm_pv_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dims of PV module of PRM')
        parser.add_argument('--prm_encoder_enc_dim', type=int, default=32, 
                            help='item encoding size of PRM')
        parser.add_argument('--prm_encoder_n_head', type=int, default=4, 
                            help='number of attention heads in transformer of PRM')
        parser.add_argument('--prm_encoder_d_forward', type=int, default=64, 
                            help='forward layer dimension in transformer of PRM')
        parser.add_argument('--prm_encoder_n_layer', type=int, default=2, 
                            help='number of encoder layers in transformer of PRM')
        parser.add_argument('--prm_pv_loss_coef', type=float, default=1.0, 
                            help='relative coefficient of pv loss')
        return parser
        
    def __init__(self, args, env, device):
        # TwoStageOnlinePolicy initialization: 
        # - initial_list_size, stage1_n_neg, stage1_state2z_hidden_dims, stage1_pos_offset, stage1_neg_offset, initial_loss_coef
        # - reader_stats, model_path, loss_type, l2_coef, no_reg, device, slate_size
        # - _define_params(args): userEncoder, enc_dim, state_dim, action_dim
        self.prm_pv_input_dim = args.prm_pv_input_dim
        self.prm_pv_hidden_dims = args.prm_pv_hidden_dims
        self.prm_encoder_enc_dim = args.prm_encoder_enc_dim
        self.prm_encoder_n_head = args.prm_encoder_n_head
        self.prm_encoder_d_forward = args.prm_encoder_d_forward
        self.prm_encoder_n_layer = args.prm_encoder_n_layer
        self.prm_pv_loss_coef = args.prm_pv_loss_coef
        super().__init__(args, env, device)
        self.display_name = "PRM"
        
    def to(self, device):
        new_self = super(PRM, self).to(device)
        new_self.PV_attn_mask = new_self.PV_attn_mask.to(device)
        new_self.PV_pos_emb_getter = new_self.PV_pos_emb_getter.to(device)
        return new_self

    def _define_params(self, args):
        '''
        Default two stage policy (pointwise initial ranker + no reranking)
        '''
        # stage1State2Z, stage1ZNorm, userEncoder, enc_dim, state_dim, action_dim
        super()._define_params(args)
        
        # input layer of PRM (personalized vector model + pos emb)
        # personalized vector model
        self.PVUserInputMap = nn.Linear(self.state_dim, args.prm_pv_input_dim)
        self.PVItemInputMap = nn.Linear(self.enc_dim, args.prm_pv_input_dim)
        self.PVInputNorm = nn.LayerNorm(args.prm_pv_input_dim)
        self.PVOutput = DNN(args.prm_pv_input_dim, args.prm_pv_hidden_dims, args.prm_encoder_enc_dim,
                            dropout_rate = args.dropout_rate, do_batch_norm = True)
        # label prediction model
        self.PVPred = nn.Linear(args.prm_encoder_enc_dim, 1)
        # positional embedding
        self.PVPosEmb = nn.Embedding(self.initial_list_size, args.prm_encoder_enc_dim)
        self.PV_pos_emb_getter = torch.arange(self.initial_list_size, dtype = torch.long)
        self.PV_attn_mask = ~torch.tril(torch.ones((self.initial_list_size,self.initial_list_size), dtype=torch.bool))
        
        # encoding layer of PRM (transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.prm_encoder_enc_dim, 
                                                   dim_feedforward = args.prm_encoder_d_forward, 
                                                   nhead=args.prm_encoder_n_head, dropout = args.dropout_rate, 
                                                   batch_first = True)
        self.PRMEncoder = nn.TransformerEncoder(encoder_layer, num_layers=args.prm_encoder_n_layer)
        
        # output layer of PRM
        self.PRMOutput = nn.Linear(args.prm_encoder_enc_dim, 1)
    
    def generate_final_action(self, user_state, feed_dict, initial_out_dict):
        '''
        @input:
        - user_state: (B, state_dim) 
        - feed_dict: same as BaseOnlinePolicy.get_forward@input-feed_dict
        - initial_out_dict: TwoStageOnlinePolicy.generate_initial_rank@output-out_dict
        @output:
        - out_dict: {
            
        }
        '''
        
        do_explore = feed_dict['do_explore']
        do_uniform = feed_dict['do_uniform']
        epsilon = feed_dict['epsilon']
        is_train = feed_dict['is_train']
        action_slate = feed_dict['action']
        
        # batch size
        B = user_state.shape[0]
        # initial list (B, C), the first K correspond to the observed slate if training
        initial_prob = initial_out_dict['initial_prob'].detach()
        initial_action = initial_out_dict['initial_action'].detach()
        candidates = initial_action
        
        # (1, L, enc_dim)
        candidate_item_emb = initial_out_dict['candidate_item_enc']
        # (B, C, enc_dim)
        initial_item_emb = candidate_item_emb.view(-1, self.enc_dim)[initial_action].detach()
        
        # input layer
        # (B, 1, pv_input_dim)
        user_input = self.PVUserInputMap(user_state).view(B,1,self.prm_pv_input_dim)
        user_input = self.PVInputNorm(user_input)
        # (B, C, pv_input_dim)
        item_input = self.PVItemInputMap(initial_item_emb.view(B*self.initial_list_size,self.enc_dim))\
                            .view(B,self.initial_list_size,self.prm_pv_input_dim)
        item_input = self.PVInputNorm(item_input)
        # (B, C, pv_input_dim)
        pv_ui_input = user_input + item_input
        # (B, C, pv_enc_dim)
        pv_ui_enc = self.PVOutput(pv_ui_input).view(B,self.initial_list_size,self.prm_encoder_enc_dim)
        # positional encoding (1, C, pv_enc_dim)
        pos_emb = self.PVPosEmb(self.PV_pos_emb_getter).view(1,self.initial_list_size,self.prm_encoder_enc_dim)
        # (B, C, pv_enc_dim)
        pv_E = pv_ui_enc + pos_emb
        
        # PRM transformer encoder output (B, C, enc_dim)
        PRM_encoder_output = self.PRMEncoder(pv_E, mask = self.PV_attn_mask)
        
        # PRM reranked score (B, C)
        rerank_score = self.PRMOutput(PRM_encoder_output.view(B*self.initial_list_size,self.prm_encoder_enc_dim))\
                                    .view(B,self.initial_list_size)
        rerank_prob = torch.softmax(rerank_score, dim = 1)
        
        if is_train or torch.is_tensor(action_slate):
            # (B, K)
            final_action = action_slate
            # (B, K)
            selected_P = rerank_prob[:,:self.slate_size]
            # label prediction (B, C)
            Y = self.PVPred(pv_E.view(B*self.initial_list_size,self.prm_encoder_enc_dim))\
                        .view(B,self.initial_list_size)
            # (B, K)
            selected_Y = Y[:,:self.slate_size]
            reg = self.get_regularization(self.PVUserInputMap, self.PVItemInputMap, self.PVOutput, 
                                          self.PVPred, self.PRMEncoder, self.PRMOutput)
            reg = reg + torch.mean(pos_emb * pos_emb)
        else:
            if do_explore:
                # exploration: categorical sampling or uniform sampling
                if do_uniform:
                    indices = Categorical(torch.ones_like(rerank_prob)).sample((self.slate_size,)).transpose(0,1)
                else:
                    indices = Categorical(rerank_prob).sample((self.slate_size,)).transpose(0,1)
            else: 
                # greedy: topk selection
                _, indices = torch.topk(rerank_prob, k = self.slate_size, dim = 1)
            indices = indices.view(-1,self.slate_size).detach()
            selected_P = torch.gather(rerank_prob,1,indices)
            final_action = torch.gather(initial_action,1,indices)
            selected_Y = None
            reg = 0
                
        
        return {'prob': selected_P, 
                'action': final_action, 
                'reward_pred': selected_Y,
                'reg': reg}

    def get_loss_observation(self):
        return ['loss', 'initial_loss', 'rerank_loss', 'pv_loss']
    
    def get_loss(self, feed_dict, out_dict):
        '''
        Reward-based pointwise ranking loss
        * - Ylog(P) - (1-Y)log(1-P)
        * Y = sum(w[i] * r[i]) # the weighted sum of user responses
        
        @input:
        - feed_dict: same as BaseOnlinePolicy.get_forward@input-feed_dict
        - out_dict: {
            'state': (B,state_dim), 
            'initial_prob': (B,K),
            'initial_action': (B,K),
            'prob': (B,K),
            'action': (B,K),
            'reward_pred': (B,K),
            'reg': scalar, 
            'immediate_response': (B,K*n_feedback),
            'immediate_response_weight: (n_feedback, ),
            'reward': (B,)}
        @output
        - loss
        '''
        B = out_dict['prob'].shape[0]
        
        # training of initial ranker
        # (B,K,n_feedback)
        weighted_response = out_dict['immediate_response'].view(B,self.slate_size,-1) \
                                * out_dict['immediate_response_weight'].view(1,1,-1)
        # (B,K)
        Y = torch.mean(weighted_response, dim = 2)
        
        if self.train_initial:
            # initial ranker loss
            initial_loss = self.get_reward_bce(out_dict['initial_prob'][:,:self.slate_size], Y)
        else:
            initial_loss = torch.tensor(0)
        
        if self.train_rerank:
            # reranker loss
            rerank_loss = self.get_reward_bce(out_dict['prob'], Y)
            # pv loss
            pv_loss = torch.mean((out_dict['reward_pred'] - Y).pow(2))
        else:
            rerank_loss, pv_loss = torch.tensor(0), torch.tensor(0)
        
        # scalar
        loss = self.initial_loss_coef * initial_loss + rerank_loss \
                    + self.prm_pv_loss_coef * pv_loss + self.l2_coef * out_dict['reg']
        
        
#         print('log(P):', torch.mean(log_P), torch.var(log_P))
#         print('log(1-P):', torch.mean(log_neg_P), torch.var(log_neg_P))
#         print('Y:', torch.mean(Y), torch.var(Y))
#         print('loss:', torch.mean(R_loss), torch.var(R_loss))
#         input()
        return {'initial_loss': loss, 
                'rerank_loss': rerank_loss, 
                'pv_loss': pv_loss, 
                'loss': loss}


        
        