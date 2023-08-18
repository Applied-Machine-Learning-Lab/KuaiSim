import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from model.components import DNN
from model.policy.BaseOnlinePolicy import BaseOnlinePolicy

class TwoStageOnlinePolicy(BaseOnlinePolicy):
    '''
    Pointwise model
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - initial_list_size
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
        parser = BaseOnlinePolicy.parse_model_args(parser)
        parser.add_argument('--initial_list_size', type=int, default=50, 
                            help='candidate list size after initial ranker')
        parser.add_argument('--stage1_state2z_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of state_slate encoding layers')
        parser.add_argument('--stage1_pos_offset', type=float, default=0.8, 
                            help='smooth offset of positive prob')
        parser.add_argument('--stage1_neg_offset', type=float, default=0.1, 
                            help='smooth offset of negative prob')
        parser.add_argument('--initial_loss_coef', type=float, default=0.1, 
                            help='relative importance of training loss of initial ranker')
        return parser
        
    def __init__(self, args, env, device):
        self.initial_list_size = args.initial_list_size
        self.stage1_state2z_hidden_dims = args.stage1_state2z_hidden_dims
        self.stage1_pos_offset = args.stage1_pos_offset
        self.stage1_neg_offset = args.stage1_neg_offset
        self.initial_loss_coef = args.initial_loss_coef
        # BaseOnlinePolicy initialization: 
        # - reader_stats, model_path, loss_type, l2_coef, no_reg, device, slate_size
        # - _define_params(args): userEncoder, enc_dim, state_dim, action_dim
        super().__init__(args, env, device)
        self.display_name = "TwoStageOnlinePolicy"
        self.train_initial = True
        self.train_rerank = True
        
    def to(self, device):
        new_self = super(TwoStageOnlinePolicy, self).to(device)
        return new_self

    def _define_params(self, args):
        '''
        Default two stage policy (pointwise initial ranker + no reranking)
        '''
        # userEncoder, enc_dim, state_dim, action_dim
        super()._define_params(args)
        # p_forward
        self.stage1State2Z = DNN(self.state_dim, args.stage1_state2z_hidden_dims, self.enc_dim, 
                           dropout_rate = args.dropout_rate, do_batch_norm = True)
        self.stage1ZNorm = nn.LayerNorm(self.enc_dim)
    
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
        # batch-wise candidates has shape (B,L), non-batch-wise candidates has shape (1,L)
        batch_wise = True
        if feed_dict['candidates']['item_id'].shape[0] == 1:
            batch_wise = False
        feed_dict['do_batch_wise'] = batch_wise
        # during training, candidates is always the full item set and has shape (1,L) where L=N
        if feed_dict['is_train']:
            assert not batch_wise
        do_uniform = np.random.random() < feed_dict['epsilon']
        feed_dict['do_uniform'] = do_uniform
        
        initial_out_dict = self.generate_initial_rank(user_state, feed_dict)
        out_dict = self.generate_final_action(user_state, feed_dict, initial_out_dict)
        out_dict['initial_prob'] = initial_out_dict['initial_prob']
        out_dict['initial_action'] = initial_out_dict['initial_action']
        out_dict['reg'] = initial_out_dict['reg'] + out_dict['reg']
        return out_dict
    
    def generate_initial_rank(self, user_state, feed_dict):
        '''
        @input:
        - user_state: (B, state_dim) 
        - feed_dict: same as BaseOnlinePolicy.get_forward@feed_dict
        @output:
        - out_dict: {'initial_prob': the initial list's item probabilities, (B, K) if training, (B, C) in inference, 
                     'initial_action': the initial list, (B, K) if training, (B, C) if inference,
                     'candidate_item_enc': (B, L, enc_dim),
                     'reg': scalar}
        '''
        candidates = feed_dict['candidates']
        slate_size = feed_dict['action_dim']
        action_slate = feed_dict['action'] # (B, K)
        do_explore = feed_dict['do_explore']
        do_uniform = feed_dict['do_uniform']
        epsilon = feed_dict['epsilon']
        is_train = feed_dict['is_train']
        batch_wise = feed_dict['do_batch_wise']
            
        B = user_state.shape[0]
        # (1,L,enc_dim) or (B,L,enc_dim)
        candidate_item_enc, reg = self.userEncoder.get_item_encoding(candidates['item_id'], 
                                                       {k[5:]: v for k,v in candidates.items() if k != 'item_id'}, 
                                                                     B if batch_wise else 1)
        # (B, enc_dim)
        Z = self.stage1State2Z(user_state)
        Z = self.stage1ZNorm(Z)
        # (B, L)
        score = torch.sum(Z.view(B,1,self.enc_dim) * candidate_item_enc, dim = -1) #/ self.enc_dim
#         score = torch.clamp(score, -self.score_clip, self.score_clip)
        
        if is_train or torch.is_tensor(action_slate):
            stage1_n_neg = self.initial_list_size - self.slate_size
            # (B, C-K)
            neg_indices = Categorical(torch.ones_like(score)).sample((stage1_n_neg,)).transpose(0,1)
            # (B, C)
            indices = torch.cat((action_slate, neg_indices), dim = 1)
            score = torch.gather(score, 1, indices)
            prob = torch.softmax(score, dim = 1)
            selected_P = prob
            initial_action = indices
            # scalar
            reg = self.get_regularization(self.stage1State2Z)
        else:
            # (B, L)
            prob = torch.softmax(score, dim = 1)
            if do_explore:
                # exploration: categorical sampling or uniform sampling
                if do_uniform:
                    indices = Categorical(torch.ones_like(prob)).sample((self.initial_list_size,)).transpose(0,1)
                else:
                    indices = Categorical(prob).sample((self.initial_list_size,)).transpose(0,1)
            else: 
                # greedy: topk selection
                _, indices = torch.topk(prob, k = self.initial_list_size, dim = 1)
            # (B, C)
            indices = indices.view(-1,self.initial_list_size).detach()
            
            selected_P = torch.gather(prob,1,indices)
            # slate action (B, K) if training or (B, C) if inference
            initial_action = indices
            reg = 0

        out_dict = {'initial_prob': selected_P, # (B, C)
                    'initial_action': initial_action, # (B, C)
                    'candidate_item_enc': candidate_item_enc, # (1, L, enc_dim)
                    'reg': reg}
        return out_dict
    
    def generate_final_action(self, user_state, feed_dict, initial_out_dict):
        '''
        @input:
        - user_state: (B, state_dim) 
        - feed_dict: same as BaseOnlinePolicy.get_forward@input-feed_dict
        - initial_out_dict: TwoStageOnlinePolicy.generate_initial_rank@output-out_dict
        @output:
        - out_dict: {
            prob: (B, K),
            action: (B, K),
            reg: scalar
        }
        '''
        
        B = user_state.shape[0]
        prob = initial_out_dict['initial_prob'][:,:self.slate_size].detach()
        slate_action = initial_out_dict['initial_action'][:,:self.slate_size].detach()
        
#         # (B, K)
#         selected_P = prob[:,:self.slate_size]
#         # (B, K)
#         initial_action = action_slate
        reg = 0
        return {'prob': prob, 
                'action': slate_action, 
                'reg': reg}

    def get_loss_observation(self):
        return ['loss', 'initial_loss', 'rerank_loss']
    
    def get_loss(self, feed_dict, out_dict):
        '''
        Reward-based pointwise ranking loss
        * - Ylog(P) - (1-Y)log(1-P)
        * Y = sum(w[i] * r[i]) # the weighted sum of user responses
        
        @input:
        - feed_dict: same as BaseOnlinePolicy.get_forward@input-feed_dict
        - out_dict: {
            'state': (B,state_dim), 
            'initial_prob': (B,C),
            'initial_action': (B,C),
            'prob': (B,K),
            'action': (B,K),
            'reg': scalar, 
            'immediate_response': (B,K*n_feedback),
            'immediate_response_weight: (n_feedback, ),
            'reward': (B,)}
        @output
        - loss
        '''
        B = out_dict['prob'].shape[0]
        # (B, K)
        initial_prob = out_dict['initial_prob'][:,:self.slate_size]
        
        if self.train_initial:
            # training of initial ranker
            # (B,K,n_feedback)
            weighted_response = out_dict['immediate_response'].view(B,self.slate_size,-1) \
                                    * out_dict['immediate_response_weight'].view(1,1,-1)
            # (B,K)
            Y = torch.mean(weighted_response, dim = 2)
            initial_loss = self.get_reward_bce(initial_prob, Y)
        else:
            initial_loss = torch.tensor(0)
        
        if self.train_rerank:
            # training of reranker
            rerank_loss = torch.zeros_like(initial_loss)
        else:
            rerank_loss = torch.tensor(0)
        
        # scalar
        loss = self.initial_loss_coef * initial_loss + rerank_loss + self.l2_coef * out_dict['reg']
        
        
#         print('log(P):', torch.mean(log_P), torch.var(log_P))
#         print('log(1-P):', torch.mean(log_neg_P), torch.var(log_neg_P))
#         print('Y:', torch.mean(Y), torch.var(Y))
#         print('loss:', torch.mean(R_loss), torch.var(R_loss))
#         input()
        return {'initial_loss': loss, 'rerank_loss': rerank_loss, 'loss': loss}

    def get_reward_bce(self, prob, y):
        # (B, K)
        log_P = torch.log(prob + self.stage1_pos_offset)
        # (B, K)
        log_neg_P = torch.log(1 - prob + self.stage1_neg_offset)
        # (B, K)
        L = - torch.mean(y * log_P + (1-y) * log_neg_P)
        return L