import numpy as np
import utils
import torch
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from reader import *
from model.simulator import *


class KREnvironment_InfiniteRec_GPU():
    '''
    KuaiRand simulated environment for list-wise recommendation
    Components:
    - multi-behavior user response model: 
        - (user history, user profile) --> user_state
        - (user_state, item) --> feedbacks (e.g. click, like, ...)
    - no user leave model:
        - user sessions stop at max_step_per_episode
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - uirm_log_path
        - slate_size
        - max_step_per_episode
        - episode_batch_size
        - item_correlation
        - env_val_holdout
        - env_test_holdout
        '''
        parser.add_argument('--uirm_log_path', type=str, required=True, 
                            help='log path for saved user immediate response model')
        parser.add_argument('--slate_size', type=int, required=6, 
                            help='number of item per recommendation slate')
        parser.add_argument('--max_step_per_episode', type=int, default=30, 
                            help='max number of iteration allowed in each episode')
        parser.add_argument('--episode_batch_size', type=int, default=128, 
                            help='episode sample batch size')
        parser.add_argument('--item_correlation', type=float, default=0.2, 
                            help='magnitude of item correlation')
        parser.add_argument('--env_val_holdout', type=int, default=0, 
                            help='val holdout')
        parser.add_argument('--env_test_holdout', type=int, default=0, 
                            help='test holdout')
        return parser
    
    def __init__(self, args):
        '''
        self.device
        self.initial_temper
        self.slate_size
        self.max_step_per_episode
        self.episode_batch_size
        self.rho
        
        self.immediate_response_stats
        self.immediate_response_model
        self.max_hist_len
        self.response_types
        self.response_dim
        self.response_weights
        
        self.reader
        self.candidate_iids
        self.candidate_item_meta
        self.n_candidate
        self.candidate_item_encoding
        self.gt_state_dim
        self.action_dim
        self.observation_space
        self.action_space
        '''
        super().__init__()
        self.device = args.device
        
        self.initial_temper = args.max_step_per_episode
        self.slate_size = args.slate_size
        self.max_step_per_episode = args.max_step_per_episode
        self.episode_batch_size = args.episode_batch_size
        self.rho = args.item_correlation
        
        print("Load immediate user response model")
        uirm_stats, uirm_model, uirm_args = self.get_user_model(args.uirm_log_path, args.device)
        self.immediate_response_stats = uirm_stats
        self.immediate_response_model = uirm_model
        self.max_hist_len = uirm_stats['max_seq_len']
        self.response_types = uirm_stats['feedback_type']
        self.response_dim = len(self.response_types)
        self.response_weights = [0 if f == 'is_hate' else 1 for f in self.response_types]
        
        print("Load user sequence reader")
        reader, reader_args = self.get_reader(args)
        self.reader = reader
        print(self.reader.get_statistics())
        
        print("Setup candiate item pool")
        # [encoded item id], size (n_item,)
        self.candidate_iids = torch.tensor([reader.item_id_vocab[iid] for iid in reader.items]).to(self.device)
        # item meta: {'if_{feature_name}': (n_item, feature_dim)}
        candidate_meta = [reader.get_item_meta_data(iid) for iid in reader.items]
        self.candidate_item_meta = {}
        self.n_candidate = len(candidate_meta)
        for k in candidate_meta[0]:
            self.candidate_item_meta[k[3:]] = torch.FloatTensor(np.concatenate([meta[k] for meta in candidate_meta]))\
                                                    .view(self.n_candidate,-1).to(self.device)
        # (n_item, item_enc_dim), groud truth encoding is implicit to RL agent
        item_enc, _ = self.immediate_response_model.get_item_encoding(self.candidate_iids, 
                                               {k:v for k,v in self.candidate_item_meta.items()}, 1)
        self.candidate_item_encoding = torch.clamp(item_enc,-1,1).view(-1,self.immediate_response_model.enc_dim)
        
        # spaces
        self.gt_state_dim = self.immediate_response_model.state_dim
        self.action_dim = self.slate_size
        self.observation_space = self.reader.get_statistics()
        self.action_space = self.n_candidate
        
        self.immediate_response_model.to(args.device)
        self.immediate_response_model.device = args.device
        
    def get_user_model(self, log_path, device, from_load = True):
        infile = open(log_path, 'r')
        class_args = eval(infile.readline()) # example: Namespace(model='KRMBUserResponse', reader='KRMBSeqReader')
        model_args = eval(infile.readline()) # model parameters in Namespace
        infile.close()
        checkpoint = torch.load(model_args.model_path + ".checkpoint", map_location=device)
        reader_stats = checkpoint["reader_stats"]
        modelClass = eval('{0}.{0}'.format(class_args.model))
        model = modelClass(model_args, reader_stats, device)
        if from_load:
            model.load_from_checkpoint(model_args.model_path, with_optimizer = False)
        model = model.to(device)
        return reader_stats, model, model_args
    
    def get_reader(self, args):
        infile = open(args.uirm_log_path, 'r')
        class_args = eval(infile.readline()) # example: Namespace(model='KRMBUserResponse', reader='KRMBSeqReader')
        training_args = eval(infile.readline()) # model parameters in Namespace
        training_args.val_holdout_per_user = args.env_val_holdout
        training_args.test_holdout_per_user = args.env_test_holdout
        training_args.device = self.device
        training_args.slate_size = args.slate_size
        infile.close()
        
        if len(args.new_reader_class) > 0:
            readerClass = eval('{0}.{0}'.format(args.new_reader_class))
        else:
            readerClass = eval('{0}.{0}'.format(class_args.reader))
        reader = readerClass(training_args)
        return reader, training_args
        
    def get_candidate_info(self, observation):
        '''
        @output:
        - candidate_info: {'item_id': (1,n_item), 
                           'item_{feature_name}': (1,n_item, feature_dim)}
        '''
        candidate_info = {'item_id': self.candidate_iids.view(1,-1)}
        candidate_info.update({f'item_{k}': v.view(1,len(self.candidate_iids),-1) \
                               for k,v in self.candidate_item_meta.items()})
        return candidate_info
    
        
    def reset(self):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': the episode running batch size, 
                    'empty_history': True if start from empty history, default = False
                    'initial_history': start with initial history, empty_history must be False}
        @output:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H, feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
        @update:
        - self.current_observation: same as @output - observation
        - self.temper
        '''
        # reset data loader
        BS = self.episode_batch_size
        self.iter = iter(DataLoader(self.reader, batch_size = BS, shuffle = True, 
                                          pin_memory = True, num_workers = 8))
        
        # initial observation
        initial_sample = next(self.iter)
        self.current_observation = self.get_observation_from_batch(initial_sample)
        self.temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
        self.user_step_count = 0
        
        return deepcopy(self.current_observation)
    
    
    def get_observation_from_batch(self, sample_batch):
        '''
        @input:
        - sample_batch: {
            'user_id': (B,)
            'uf_{feature}': (B,F_dim(feature)), user features
            'history': (B,max_H)
            'history_length': (B,)
            'history_if_{feature}': (B, max_H * F_dim(feature))
            'history_{response}': (B, max_H)
            ... other unrelated features
        }
        @output:
        - observation: same as self.reset@output - observation
        '''
        sample_batch = utils.wrap_batch(sample_batch, device = self.device)
        profile = {'user_id': sample_batch['user_id']}
        for k,v in sample_batch.items():
            if 'uf_' in k:
                profile[k] = v
        history = {'history': sample_batch['history']}
        for k,v in sample_batch.items():
            if 'history_' in k:
                history[k] = v
        return {'user_profile': profile, 'user_history': history}
    
    def step(self, step_dict):
        '''
        @input:
        - step_dict: {'action': (B, action_dim)}  Note: action must be indices on candidate_iids
        @output:
        - new_observation (may not be the same user)
        - user_feedback: {'immediate_response': (B, slate_size, n_feedback), 
                          'done': (B,),
                          'coverage': scalar,
                          'ILD': scalar}
        - updated_observation (next observation of the same user)
        '''
        
        # (B, slate_size)
        action = step_dict['action']
        
        # user interaction
        with torch.no_grad():
            response_out = self.get_response(step_dict)
            # (B, slate_size, n_feedback)
            response = response_out['immediate_response']

            # get leave signal
            # (B,), 0-1 vector
            done_mask = self.get_leave_signal(response)
            
            # update observation
            update_info = self.update_observation(action, response, done_mask)
            self.user_step_count += 1
            
            for i,f in enumerate(self.response_types):
                # (B, )
                R = response.mean(1)[:,i].detach()

            if done_mask.sum() == len(done_mask):
                new_iter_flag = False
                try:
                    sample_info = next(self.iter)
                    if sample_info['user_profile'].shape[0] != len(done_mask):
                        new_sample_flag = True
                except:
                    new_sample_flag = True
                if new_sample_flag:
                    self.iter = iter(DataLoader(self.reader, batch_size = done_mask.shape[0], shuffle = True, 
                                                pin_memory = True, num_workers = 8))
                    sample_info = next(self.iter)
                new_observation = self.get_observation_from_batch(sample_info)
                self.current_observation = new_observation
                self.temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
                self.user_step_count = 0
            elif done_mask.sum() > 0:
                print(done_mask)
                print("User leave not synchronized")
                raise NotImplemented
        user_feedback = {'immediate_response': response, 
                         'done': done_mask, 
                         'coverage': response_out['coverage'], 
                         'ILD': response_out['ILD']}
        return deepcopy(self.current_observation), user_feedback, update_info['updated_observation']
    
    def get_response(self, step_dict):
        '''
        @input:
        - step_dict: {'action': (B, slate_size)}
        @output:
        - response: (B, slate_size, n_feedback), 0-1 tensor
        '''
        # (B, slate_size)
        action = step_dict['action']
        coverage = len(torch.unique(action))
        B = action.shape[0]
        
        # (B, 1, gt_state_dim)
        profile_dict = {k:v for k,v in self.current_observation['user_profile'].items()}
        user_state = self.get_ground_truth_user_state(profile_dict,
                                        {k:v for k,v in self.current_observation['user_history'].items()})
        user_state = user_state.view(self.episode_batch_size, 1, self.gt_state_dim)

        ########################################
        # This is where the action take effect #
        # (B, slate_size, item_enc_dim)
        selected_item_enc = self.candidate_item_encoding[action].view(B, self.slate_size, 1,
                                                                      self.immediate_response_model.enc_dim)
        ########################################

        # point-wise model gives xtr scores
        # (B, slate_size, n_feedback), _
        behavior_scores, _ = self.immediate_response_model.get_pointwise_scores(user_state, selected_item_enc, 
                                                                                self.episode_batch_size)


        # item correlation: assuming users always wants more diverse item list
        # (B, slate_size)
        corr_factor = self.get_intra_slate_similarity(selected_item_enc.view(B, self.slate_size, -1))
        # (B, slate_size, n_feedback)
        point_scores = torch.sigmoid(behavior_scores) - corr_factor.view(B, self.slate_size, 1) * self.rho
        point_scores[point_scores < 0] = 0
        
        # user response sampling
        # (B, slate_size, n_feedback)
        response = torch.bernoulli(point_scores)
        
        return {'immediate_response': response, 
                'coverage': coverage,
                'ILD': 1 - torch.mean(corr_factor).item()}

    def get_ground_truth_user_state(self, profile, history):
        batch_data = {}
        batch_data.update(profile)
        batch_data.update(history)
        gt_state_dict = self.immediate_response_model.encode_state(batch_data, self.episode_batch_size)
        gt_user_state = gt_state_dict['state'].view(self.episode_batch_size,1,self.gt_state_dim)
        return gt_user_state
    
    def get_intra_slate_similarity(self, action_item_encoding):
        '''
        @input:
        - action_item_encoding: (B, slate_size, enc_dim)
        @output:
        - similarity: (B, slate_size)
        '''
        B, L, d = action_item_encoding.shape
        # pairwise similarity in a slate (B, L, L)
        pair_similarity = torch.mean(action_item_encoding.view(B,L,1,d) * action_item_encoding.view(B,1,L,d), dim = -1)
        # similarity to slate average, (B, L)
        point_similarity = torch.mean(pair_similarity, dim = -1)
        return point_similarity
        

    def update_observation(self, action, slate_response, done_mask):
        '''
        @input:
        - action: (B, slate_size)  Note: action must be indices on candidate_iids
        - slate_response: (B, slate_size, n_feedback)
        - done_mask: (B,)
        
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H * feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
        '''
        # (B, slate_size)
        rec_list = self.candidate_iids[action]
        
        old_history = self.current_observation['user_history']
        max_H = self.max_hist_len
        L = old_history['history_length'] + self.slate_size
        L[L>max_H] = max_H
        new_history = {'history': torch.cat((old_history['history'], rec_list), dim = 1)[:,-max_H:], 
                       'history_length': L}
        for f in self.reader.selected_item_features:
            # (n_item, feature_dim)
            candidate_meta_features = self.candidate_item_meta[f]
            # (B, slate_size, feature_dim)
            meta_features = candidate_meta_features[action]
            k = f'history_if_{f}'
            # (B, max_H, feature_dim)
            previous_meta = old_history[k].view(self.episode_batch_size, self.observation_space['max_seq_len'], -1)
            new_history[k] = torch.cat((previous_meta, meta_features), dim = 1)[:,-max_H:,:].view(self.episode_batch_size,-1)
        for i,response in enumerate(self.immediate_response_model.feedback_types):
            k = f'history_{response}'
            new_history[k] = torch.cat((old_history[k], slate_response[:,:,i]), dim = 1)[:,-max_H:]
        self.current_observation['user_history'] = new_history
        return {'slate': rec_list, 'updated_observation': deepcopy(self.current_observation)}

    def get_leave_signal(self, response):
        '''
        @input:
        - user_state: (B, state_dim)
        - response: (B, slate_size, n_feedback)
        '''
        temper_down = 1
        self.temper -= temper_down
        done_mask = self.temper < 1
        return done_mask
        
    def create_observation_buffer(self, buffer_size):
        '''
        @output:
        - observation: {'user_profile': {'user_id': (L,), 
                                         'uf_{feature_name}': (L, feature_dim)}, 
                        'user_history': {'history': (L, max_H), 
                                         'history_if_{feature_name}': (L, max_H * feature_dim), 
                                         'history_{response}': (L, max_H), 
                                         'history_length': (L,)}}
        '''
        observation = {'user_profile': {'user_id': torch.zeros(buffer_size).to(torch.long).to(self.device)}, 
                       'user_history': {'history': torch.zeros(buffer_size, self.max_hist_len).to(torch.long).to(self.device), 
                                        'history_length': torch.zeros(buffer_size).to(torch.long).to(self.device)}}
        for f,f_dim in self.observation_space['user_feature_dims'].items():
            observation['user_profile'][f'uf_{f}'] = torch.zeros(buffer_size, f_dim).to(torch.float).to(self.device)
        for f,f_dim in self.observation_space['item_feature_dims'].items():
            observation['user_history'][f'history_if_{f}'] = torch.zeros(buffer_size, f_dim * self.max_hist_len)\
                                                                                .to(torch.float).to(self.device)
        for f in self.observation_space['feedback_type']:
            observation['user_history'][f'history_{f}'] = torch.zeros(buffer_size, self.max_hist_len)\
                                                                                .to(torch.float).to(self.device)
        return observation
    
        
    def stop(self):
        self.iter = None
    
    def get_new_iterator(self, B):
        return iter(DataLoader(self.reader, batch_size = B, shuffle = True, 
                               pin_memory = True, num_workers = 8))
        
    def get_env_report(self, window = 50):
        return {}
        
        
        