import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np

from model.general import BaseModel
from model.components import DNN
from model.policy.BaseOnlinePolicy import BaseOnlinePolicy

class ListCVAE(BaseOnlinePolicy):
    '''
    GFlowNet with Trajectory Balance for listwise recommendation
    
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - cvae_c_dim
        - cvae_z_dim
        - cvae_prior_hidden_dims
        - cvae_encoder_hidden_dims
        - cvae_decoder_hidden_dims
        - cvae_beta
        - target_reward
        - from BaseOnlinePolicy:
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
        parser.add_argument('--cvae_c_dim', type=int, default=32, 
                            help='embedding size of condition')
        parser.add_argument('--cvae_z_dim', type=int, default=32, 
                            help='latent embedding size')
        parser.add_argument('--cvae_prior_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of state_slate encoding layers')
        parser.add_argument('--cvae_encoder_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of encoder')
        parser.add_argument('--cvae_decoder_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of decoder')
        parser.add_argument('--cvae_beta', type=float, default=0.1, 
                            help='trade-off coefficient between reconstruction and KLD loss')
        parser.add_argument('--target_reward', type=float, default=3, 
                            help='target reward during inference')
        parser.add_argument('--n_neg', type=int, default=100, 
                            help='number of negative samples in sampled softmax')
        
        return parser
        
    def __init__(self, args, reader_stats, device):
        # BaseModel initialization: 
        # - reader_stats, model_path, loss_type, l2_coef, no_reg, device, slate_size
        # - _define_params(args): enc_dim, state_dim, action_dim
        self.c_dim = args.cvae_c_dim
        self.z_dim = args.cvae_z_dim
        self.target_reward = args.target_reward
        self.beta = args.cvae_beta
        self.n_neg = args.n_neg
        super().__init__(args, reader_stats, device)
        self.display_name = "ListCVAE"
        self.CEL = nn.CrossEntropyLoss()
        
    def to(self, device):
        new_self = super(ListCVAE, self).to(device)
        return new_self

    def _define_params(self, args):
        # userEncoder, enc_dim, state_dim, bce_loss
        super()._define_params(args)
        # encoder
        self.encoder = DNN(self.slate_size * self.enc_dim + self.c_dim + self.state_dim,
                           args.cvae_encoder_hidden_dims, self.z_dim * 2, 
                           dropout_rate = args.dropout_rate, do_batch_norm = True)
        self.zNorm = nn.LayerNorm(self.z_dim)
        
        # decoder
        self.decoder = DNN(self.z_dim + self.c_dim + self.state_dim,
                           args.cvae_decoder_hidden_dims, self.slate_size * self.enc_dim, 
                           dropout_rate = args.dropout_rate, do_batch_norm = True)
        
        # prior
        self.prior = DNN(self.c_dim + self.state_dim, 
                         args.cvae_prior_hidden_dims, self.z_dim * 2, 
                         dropout_rate = args.dropout_rate, do_batch_norm = True)
        
        # response condition vector
        self.C = nn.Linear(1, self.c_dim)
        self.CNorm = nn.LayerNorm(self.c_dim)
    
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
        - user_state: (B, state_dim) 
        - feed_dict: same as BaseOnlinePolicy.get_forward@feed_dict
        @output:
        - out_dict: {'logP': (B, K), 
                     'logF0': (B,),
                     'action': (B, K), 
                     'reg': scalar}
        '''
        
        candidates = feed_dict['candidates']
        slate_size = feed_dict['action_dim']
        slate_action = feed_dict['action'] # (B, K)
        slate_response = feed_dict['response'] # {'reward': (B,), 'immediate_response': (B,K*n_feedback)}
        do_explore = feed_dict['do_explore']
        is_train = feed_dict['is_train']
        epsilon = feed_dict['epsilon']
        B = user_state.shape[0]
        reg = 0
        # batch-wise candidates has shape (B,L), non-batch-wise candidates has shape (1,L)
        batch_wise = True
        if candidates['item_id'].shape[0] == 1:
            batch_wise = False
        # during training, candidates is always the full item set and has shape (1,L) where L=N
        if is_train:
            assert not batch_wise
        # epsilon probability for uniform sampling under exploration
        do_uniform = np.random.random() < epsilon
            
        # (1,L,enc_dim)
        candidate_item_enc, candidate_reg = self.userEncoder.get_item_encoding(candidates['item_id'], 
                                                       {k[5:]: v for k,v in candidates.items() if k != 'item_id'}, 
                                                                     B if batch_wise else 1)
        
        if is_train:
            # Training uses the encoder: (slate, condition, user) --> z
            # condition vector (B, c_dim)
            if 'reward' in feed_dict:
                cond_vec = self.C(feed_dict['reward'].view(B,1)).view(B, self.c_dim)
            else:
                cond_vec = self.C(slate_response['reward'].view(B,1)).view(B, self.c_dim)
            cond_vec = self.CNorm(cond_vec)
            # input slate (B, K, enc_dim)
            slate_item_enc = candidate_item_enc.view(-1,self.enc_dim)[slate_action]
            # (B, K*enc_dim)
            slate_item_enc = slate_item_enc.view(B,-1)
            # encode
            z_mu, z_logvar, enc_reg = self.encode(slate_item_enc, cond_vec, user_state)
            # prior
            prior_z_mu, prior_z_logvar, prior_reg = self.get_prior(cond_vec, user_state)
            reg = enc_reg + candidate_reg + prior_reg + reg
        else:
            # Inference uses the prior: (condition, user) --> z
            # condition vector (B, c_dim)
            cond_vec = self.C(torch.ones(B,1).to(self.device)*self.target_reward).view(B, self.c_dim)
            cond_vec = self.CNorm(cond_vec)
            # prior
            prior_z_mu, prior_z_logvar, prior_reg = self.get_prior(cond_vec, user_state)
            z_mu, z_logvar = prior_z_mu, prior_z_logvar
            
        # (B, z_dim)
        z = self.reparametrize(z_mu, z_logvar)
        # (B, K, enc_dim)
        output_slate_emb, dec_reg = self.decode(z, cond_vec, user_state)
        # (B, K, 1, enc_dim)
        output_slate_emb = output_slate_emb.view(B, self.slate_size, 1, self.enc_dim)
        
        if is_train or torch.is_tensor(slate_action):
            candidate_item_enc = candidate_item_enc.view(-1,self.enc_dim)
            L = candidate_item_enc.shape[0]
            # sampled negative scores for better efficiency
            # (n_neg,)
            sampled_neg = torch.randint(0, L, (self.n_neg,))
            # (1, 1, n_neg, enc_dim)
            sampled_neg_enc = candidate_item_enc[sampled_neg].view(1,1,self.n_neg,self.enc_dim)
            # (B, K, n_neg)
            sampled_neg_scores = torch.mean(output_slate_emb * sampled_neg_enc, dim = -1)
            # positive item scores
            # (B, K, 1, enc_dim)
            pos_enc = candidate_item_enc[slate_action].view(B,self.slate_size,1,self.enc_dim)
            # (B, K, 1)
            pos_scores = torch.mean(output_slate_emb * pos_enc, dim = -1).view(B,self.slate_size,1)
            # (B, K, n_neg + 1)
            item_prob = torch.cat((pos_scores, sampled_neg_scores), dim = 2)
            item_prob = torch.softmax(item_prob, dim = 2)
            # (B, K)
            selected_prob = item_prob[:,:,0]
            # (B, K)
            output_action = slate_action
            reg = dec_reg + reg
        else:
            # (B, K, L)
            item_prob = torch.mean(output_slate_emb * candidate_item_enc, dim = -1)
            item_prob = torch.softmax(item_prob, dim = -1)
            if do_explore:
                # (B, K)
                if do_uniform:
                    output_action = Categorical(torch.ones_like(item_prob)).sample()
                else:
                    output_action = Categorical(item_prob).sample()
                # (B, K)
                selected_prob = torch.gather(item_prob, 2, output_action.view(B,self.slate_size,1))
            else:
                # (B, K)
                selected_prob, output_action = torch.topk(item_prob, 1)
            reg = 0
        

        out_dict = {'prob': selected_prob.view(B, self.slate_size),
                    'all_prob': item_prob,
                    'action': output_action.view(B, self.slate_size), 
                    'x_prime': output_slate_emb.view(B, self.slate_size, self.enc_dim),
                    'z_mu': z_mu,
                    'z_logvar': z_logvar,
                    'prior_z_mu': prior_z_mu,
                    'prior_z_logvar': prior_z_logvar,
                    'reg': reg}
        return out_dict
    
    def encode(self, S, C, U):
        '''
        @input:
        - S: (B, slate_size, enc_dim)
        - C: (B, c_dim)
        - U: (B, state_dim)
        @output:
        - mu: (B, z_dim)
        - logvar: (B, z_dim)
        - reg: scalar
        '''
        B = U.shape[0]
        # (B, slate_size * enc_dim + c_dim + state_dim)
        X = torch.cat((S.view(B,-1), C.view(B,-1), U.view(B,-1)), dim = 1)
        # (B, z_dim * 2)
        z_output = self.encoder(X).view(B, 2, self.z_dim)
        z_output = self.zNorm(z_output)
        # (B, z_dim)
        z_mu = z_output[:,0,:].view(B, self.z_dim)
        z_logvar = z_output[:,1,:].view(B, self.z_dim)
        # scalar
        reg = self.get_regularization(self.encoder)
        return z_mu, z_logvar, reg
    
    def get_prior(self, C, U):
        '''
        @input:
        - C: (B, c_dim)
        - U: (B, state_dim)
        @output:
        - mu: (B, z_dim)
        - logvar: (B, z_dim)
        - reg: scalar
        '''
        B = U.shape[0]
        # (B, c_dim + state_dim)
        X = torch.cat((C.view(B,-1), U.view(B,-1)), dim = 1)
        # (B, z_dim * 2)
        z_output = self.prior(X).view(B, 2, self.z_dim)
        z_output = self.zNorm(z_output)
        # (B, z_dim)
        z_mu = z_output[:,0,:].view(B, self.z_dim)
        z_logvar = z_output[:,1,:].view(B, self.z_dim)
        # scalar
        reg = self.get_regularization(self.prior)
        return z_mu, z_logvar, reg
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        z = eps.mul(std) + mu
        return z
    
    def decode(self, Z, C, U):
        '''
        @input:
        - Z: (B, z_dim
        - C: (B, c_dim)
        - U: (B, state_dim)
        @output:
        - mu: (B, z_dim)
        - logvar: (B, z_dim)
        - reg: scalar
        '''
        B = U.shape[0]
        # (B, z_dim + c_dim + state_dim)
        latent = torch.cat((Z.view(B,-1), C.view(B,-1), U.view(B,-1)), dim = 1)
        # (B, slate_size * enc_dim)
        X_prime = self.decoder(latent).view(B, self.slate_size, self.enc_dim)
        # scalar
        reg = self.get_regularization(self.decoder)
        return X_prime, reg
    
    def get_loss(self, feed_dict, out_dict):
        '''
        Trajectory balance loss
        @input:
        - feed_dict: same as BaseOnlinePolicy.get_forward@input-feed_dict
        - out_dict: {
            'state': (B,state_dim), 
            'prob': (B,K),
            'all_prob': (B,L) # L is the item pool size
            'action': (B,K),
            'x_prime': (B,K,enc_dim),
            'z_mu': (B, z_dim),
            'z_logvar': (B, z_dim),
            'prior_z_mu': (B, z_dim),
            'prior_z_logvar': (B, z_dim),
            'reg': scalar, 
            'immediate_response': (B,K*n_feedback),
            'immediate_response_weight: (n_feedback, ),
            'reward': (B,)}
        @output
        - loss
        '''
        B = out_dict['prob'].shape[0]
        # log likelihood maximization
        rec_loss = - torch.mean(torch.log(out_dict['prob']))
#         # cross entropy loss
#         rec_loss = self.CEL(out_dict['all_prob'].view(B*self.slate_size,-1), out_dict['action'].view(-1))
        # KLD loss
        mu, logvar, pMu, pLogvar = out_dict['z_mu'], out_dict['z_logvar'], out_dict['prior_z_mu'], out_dict['prior_z_logvar']
        KLD = - 0.5 * torch.mean(1 + logvar - pLogvar - (logvar.exp() + (mu - pMu).pow(2)) / pLogvar.exp())
        # ELBO
        loss = rec_loss + self.beta * KLD + self.l2_coef * out_dict['reg']
#         print('rec_loss:', torch.mean(rec_loss), torch.var(rec_loss))
#         print('KLD:', torch.mean(KLD), torch.var(KLD))
#         print('loss:', torch.mean(loss), torch.var(loss))
#         input()
        return {'loss': loss, 'rec_loss': rec_loss, 'KLD': KLD}

        
    def get_loss_observation(self):
        return ['loss', 'rec_loss', 'KLD']
        
        