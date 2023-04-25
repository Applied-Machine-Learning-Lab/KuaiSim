import torch
import numpy as np

from model.policy.OneStagePolicy import OneStagePolicy
from model.components import DNN
from model.score_func import *

class OneStageHyperPolicy_with_DotScore(OneStagePolicy):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - policy_action_hidden
        - policy_noise_var
        - policy_noise_clip
        - policy_do_effect_action_explore
        - from OneStagePolicy:
            - from BackboneUserEncoder:
                - state_user_latent_dim
                - state_item_latent_dim
                - state_transformer_enc_dim
                - state_transformer_n_head
                - state_transformer_d_forward
                - state_transformer_n_layer
                - state_dropout_rate
                - from BaseModel:
                    - model_path
                    - loss
                    - l2_coef
        '''
        parser = OneStagePolicy.parse_model_args(parser)
        parser.add_argument('--policy_action_hidden', type=int, nargs='+', default=[128], 
                            help='hidden dim of the action net')
        parser.add_argument('--policy_noise_var', type=float, default=0.1, 
                            help='hyper action exploration variance')
        parser.add_argument('--policy_noise_clip', type=float, default=1.0, 
                            help='hyper action clip bound')
        parser.add_argument('--policy_do_effect_action_explore', action='store_true',
                            help='do exploration on effect action')
        return parser
    
    def __init__(self, *input_args):
        '''
        components:
        - user_encoder
        - hyper_action_layer
        - state_dim, enc_dim, action_dim
        '''
        args, env = input_args
        self.noise_var = args.policy_noise_var
        self.noise_clip = args.policy_noise_clip
        self.do_effect_action_explore = args.policy_do_effect_action_explore
        super().__init__(args, env)
        # action is the set of parameters of linear mapping [item_dim + 1, 1]
        self.hyper_action_dim = self.enc_dim + 1
        self.action_dim = self.hyper_action_dim
        self.effect_action_dim = self.slate_size
        self.hyper_action_layer = DNN(self.state_dim, args.policy_action_hidden, self.hyper_action_dim, 
                                      dropout_rate = self.dropout_rate, do_batch_norm = True)
        
    def generate_action(self, state_dict, feed_dict):
        '''
        List generation provides three main types of exploration:
        * Greedy top-K: no exploration
        * Categorical sampling: probabilistic exploration
        * Uniform sampling: random exploration
        
        @input:
        - state_dict: {'state': (B, state_dim), ...}
        - feed_dict: same as OneStagePolicy.get_forward@input - feed_dict
        @output:
        - out_dict: {'preds': (B, K), 
                     'action': (B, hyper_action_dim), 
                     'indices': (B, K),
                     'hyper_action': (B, hyper_action_dim),
                     'effect_action': (B, K),
                     'all_preds': (B, L),
                     'reg': scalar}
        '''
        user_state = state_dict['state'] # (B, state_dim)
        candidates = feed_dict['candidates']
        epsilon = feed_dict['epsilon']
        do_explore = feed_dict['do_explore']
        is_train = feed_dict['is_train']
        batch_wise = feed_dict['batch_wise']
        
        B = user_state.shape[0]
        # epsilon probability for uniform sampling under exploration
        do_uniform = np.random.random() < epsilon
        # (B, hyper_action_dim)
        hyper_action_raw = self.hyper_action_layer(user_state).view(B, self.action_dim)
#         print('hyper_action_raw:', hyper_action_raw.shape)
        
        # (B, hyper_action_dim), hyper action exploration
        if do_explore:
            if do_uniform:
                hyper_action = torch.clamp(torch.rand_like(hyper_action_raw)*self.noise_var, 
                                           -self.noise_clip, self.noise_clip)
            else:
                hyper_action = hyper_action_raw + torch.clamp(torch.rand_like(hyper_action_raw)*self.noise_var, 
                                           -self.noise_clip, self.noise_clip)
        else:
            hyper_action = hyper_action_raw
        
        # (B, L, enc_dim) if batch_wise candidates, otherwise (1,L,enc_dim)
        candidate_item_enc, reg = self.user_encoder.get_item_encoding(candidates['item_id'], 
                                                       {k[3:]: v for k,v in candidates.items() if k != 'item_id'}, 
                                                                     B if batch_wise else 1)
#         print('candidate_item_enc:', candidate_item_enc.shape)
        # (B, L)
        scores = self.get_score(hyper_action, candidate_item_enc, self.enc_dim)
#         print('scores:', scores.shape)
        
        # effect action exploration in both training and inference
        if self.do_effect_action_explore and do_explore:
            if do_uniform:
                # categorical sampling
                action, indices = utils.sample_categorical_action(P, candidates['item_id'], 
                                                                  self.slate_size, with_replacement = False, 
                                                                  batch_wise = batch_wise, return_idx = True)
            else:
                # uniform sampling happens only in inference time
                action, indices = utils.sample_categorical_action(torch.ones_like(P), candidates['item_id'], 
                                                                  self.slate_size, with_replacement = False, 
                                                                  batch_wise = batch_wise, return_idx = True)
        else:
            # top-k selection
            _, indices = torch.topk(scores, k = self.slate_size, dim = 1)
            if batch_wise:
                action = torch.gather(candidates['item_id'], 1, indices).detach() # (B, slate_size)
            else:
                action = candidates['item_id'][indices].detach() # (B, slate_size)
#         print('action:', action.shape)
#         print(action)
#         input()
        action_scores = torch.gather(scores, 1, indices).detach()
        
        reg = reg + self.get_regularization(self.hyper_action_layer)

        out_dict = {'preds': action_scores,
                    'action': hyper_action, 
                    'indices': indices,
                    'hyper_action': hyper_action,
                    'effect_action': action,
                    'all_preds': scores,
                    'reg': reg}
        return out_dict
    
    def get_score(self, hyper_action, candidate_item_enc, item_dim):
        '''
        Deterministic mapping from hyper-action to effect-action (rec list)
        '''
        # (B, L)
        scores = linear_scorer(hyper_action, candidate_item_enc, item_dim)
        return scores
    
    def forward(self, feed_dict: dict, return_prob = True):
        out_dict = self.get_forward(feed_dict)
        if return_prob:
            out_dict['all_probs'] = torch.softmax(out_dict['all_preds'], dim = 1)
            out_dict['probs'] = torch.gather(out_dict['all_probs'], 1, out_dict['indices'])
        return out_dict
    