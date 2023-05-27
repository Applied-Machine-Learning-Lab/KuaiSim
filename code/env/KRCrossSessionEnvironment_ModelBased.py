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


class KRCrossSessionEnvironment_ModelBased():
    '''
    KuaiRand simulated environment on GPU machines
    Components:
    - multi-behavior user response model: 
        - (user history, user profile) --> user_state
        - (user_state, item) --> feedbacks (e.g. click, long_view, like, ...)
    - user leave model:
        - user temper reduces to <1 and leave
        - user temper drops gradually through time and further drops when the user is unsatisfactory about a recommendation
    - user retention model:
        - [end of session user_state] --> user retention
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - uirm_log_path
        - urrm_log_path
        - max_n_session
        - initial_temper
        '''
        parser.add_argument('--uirm_log_path', type=str, required=True, 
                            help='log path for saved user immediate response model')
        parser.add_argument('--urrm_log_path', type=str, required=True, 
                            help='log path for saved user retention response model')
        parser.add_argument('--max_n_session', type=int, default=30, 
                            help='max number of sessions per user episode')
        parser.add_argument('--initial_temper', type=int, required=10, 
                            help='initial temper of users')
        parser.add_argument('--slate_size', type=int, required=6, 
                            help='number of item per recommendation slate')
        return parser
    
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        
        self.max_n_session = args.max_n_session
        self.initial_temper = args.initial_temper
        self.slate_size = args.slate_size
        
        print("Load immediate user response model")
        uirm_stats, uirm_model, uirm_args = self.get_user_model(args.uirm_log_path, args.device)
        self.immediate_response_stats = uirm_stats
        self.immediate_response_model = uirm_model
        self.max_hist_len = uirm_stats['max_seq_len']
        
        print("Load user retention model")
        urrm_stats, urrm_model, urrm_args = self.get_user_model(args.urrm_log_path, args.device, from_load = False)
        self.retention_stats = urrm_stats
        self.retention_model = urrm_model
        
        print("Load user retention reader")
        reader, reader_args = self.get_reader(args.uirm_log_path)
        self.reader = reader
        
        print("Setup candiate item pool")
        # [encoded item id], size (n_item,)
        self.candidate_iids = torch.tensor([reader.item_id_vocab[iid] for iid in reader.items]).to(self.device)
        # item meta: {'if_{feature_name}': (n_item, feature_dim)}
        candidate_meta = [reader.get_item_meta_data(iid) for iid in reader.items]
        self.candidate_item_meta = {}
        self.n_candidate = len(candidate_meta)
        for k in candidate_meta[0]:
            self.candidate_item_meta[k] = torch.FloatTensor(np.concatenate([meta[k] for meta in candidate_meta]))\
                                                    .view(self.n_candidate,-1).to(self.device)
        item_enc, _ = self.immediate_response_model.get_item_encoding(self.candidate_iids, 
                                               {k[3:]:v for k,v in self.candidate_item_meta.items()}, 1)
        # (n_item, item_enc_dim), groud truth encoing is implicit to RL agent
        self.candidate_item_encoding = item_enc.view(-1,self.immediate_response_model.enc_dim)
        
        # spaces
        self.gt_state_dim = self.immediate_response_model.state_dim
        self.action_dim = self.immediate_response_model.feedback_dim
        self.observation_space = self.reader.get_statistics()
        
        self.device = args.device
        self.immediate_response_model.to(args.device)
        self.immediate_response_model.device = args.device
        self.retention_model.to(args.device)
        self.retention_model.device = args.device
        
        
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
    
    def get_reader(self, log_path):
        infile = open(log_path, 'r')
        class_args = eval(infile.readline()) # example: Namespace(model='KRMBUserResponse', reader='KRMBSeqReader')
        training_args = eval(infile.readline()) # model parameters in Namespace
        training_args.val_holdout_per_user = 0
        training_args.test_holdout_per_user = 0
        infile.close()
        readerClass = eval('{0}.{0}'.format(class_args.reader))
        reader = readerClass(training_args)
        return reader, training_args
        
    def reset(self, params = {'batch_size': 1, 'empty_history': True}):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': scalar, 
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
        - self.user_leave_history
        - self.return_history
        - self.temper
        '''
        self.empty_history_flag = params['empty_history'] if 'empty_history' in params else False
        BS = params['batch_size']
        self.episode_batch_size = BS
        self.batch_iter = iter(DataLoader(self.reader, batch_size = BS, shuffle = True, 
                                          pin_memory = True, num_workers = 8))
        initial_sample = next(self.batch_iter)
        self.current_observation = self.get_observation_from_batch(initial_sample)
        
        self.user_leave_history = []
        self.user_return_history = [np.array([self.action_dim]*self.episode_batch_size)]
        self.user_total_return_gap = [np.array([self.action_dim*self.max_n_session]*self.episode_batch_size)]
        self.temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
        self.session_count = 0
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
        - step_dict: {'action': (B, W_dim)}
        '''
        # actions (exposures)
        action = step_dict['action'] # (B, W_dim), the fusion weight
        
        # user interaction
        with torch.no_grad():
            # point-wise model gives xtr scores
            # (B, 1, state_dim)
            profile_dict = {k:v for k,v in self.current_observation['user_profile'].items()}
            user_state = self.get_ground_truth_user_state(profile_dict,
                                            {k:v for k,v in self.current_observation['user_history'].items()})
            user_state = user_state.view(self.episode_batch_size, 1, self.gt_state_dim)
            # (B, n_item, n_feedback), _
            behavior_scores, _ = self.immediate_response_model.get_pointwise_scores(user_state, 
                                                    self.candidate_item_encoding[None,:,None,:], 
                                                    self.episode_batch_size)
            
            # generate recommendation list
            ########################################
            # This is where the action take effect #
            # (B, n_item, n_feedback)
            point_scores = torch.sigmoid(behavior_scores)
            # (B, n_item)
            ranking_scores = torch.sum(point_scores * action.view(self.episode_batch_size,1,self.action_dim), dim = -1)
            ########################################
            
            # _, (B, slate_size)
            _, indices = torch.topk(ranking_scores, self.slate_size, dim = -1)
            
            # sample immediate responses
            # (B, slate_size, n_feedback)
            score_indices = torch.tile(indices[:,:,None], (1,1,point_scores.shape[-1]))
            selected_scores = torch.gather(point_scores, 1, score_indices) 
            # (B, slate_size, n_feedback)
            response = torch.bernoulli(selected_scores)

            # get leave signal
            # (B,), 0-1 vector
            done_mask = self.get_leave_signal(response)
            self.user_leave_history.append(done_mask.detach().cpu().numpy())
            
            # update observation
            update_info = self.update_observation(indices, response, done_mask)
            
            # get retention signal
            new_gt_state = self.get_ground_truth_user_state(profile_dict, 
                                                            update_info['updated_observation']['user_history'])
            retention_out_dict = self.retention_model({'user_id': self.current_observation['user_profile']['user_id'], 
                                                       'sess_encoding': new_gt_state})
            # (B, max_return_day)
            retention_prob = torch.softmax(retention_out_dict['preds'], dim = -1)
            # (B,)
#             print('retention_prob:')
#             print(retention_prob)
#             input()
            return_day = Categorical(retention_prob).sample() + 1
            return_day = return_day * done_mask # has return signal when leaving current session

            if done_mask.sum() == len(done_mask):
                # refresh temper in every session
                self.temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
                self.user_return_history.append(return_day.detach().cpu().numpy())
                self.session_count += 1
                # refresh users when the last session is done 
                if self.session_count == self.max_n_session:
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
                    self.user_total_return_gap.append(np.sum(np.array(self.user_return_history[-self.max_n_session:]), axis = 0))
                    self.session_count = 0
            elif done_mask.sum() > 0:
                print(done_mask)
                print("Leaving not synchronized")
                raise NotImplemented
        user_feedback = {'immediate_response': response, 
                         'done': done_mask, 
                         'retention': return_day.to(torch.float)}
        return deepcopy(self.current_observation), user_feedback, update_info['updated_observation']

    def get_ground_truth_user_state(self, profile, history):
        batch_data = {}
        batch_data.update(profile)
        batch_data.update(history)
        gt_state_dict = self.immediate_response_model.encode_state(batch_data, self.episode_batch_size)
        gt_user_state = gt_state_dict['state'].view(self.episode_batch_size,1,self.gt_state_dim)
        return gt_user_state

    def update_observation(self, slate_indices, slate_response, done_mask):
        '''
        @input:
        - slate_indices: (B, slate_size)
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
        rec_list = self.candidate_iids[slate_indices]
        
        old_history = self.current_observation['user_history']
        max_H = self.max_hist_len
        L = old_history['history_length'] + self.slate_size
        L[L>max_H] = max_H
        new_history = {'history': torch.cat((old_history['history'], rec_list), dim = 1)[:,-max_H:], 
                       'history_length': L}
        for f in self.reader.selected_item_features:
            # (n_item, feature_dim)
            candidate_meta_features = self.candidate_item_meta[f'if_{f}']
            # (B, slate_size, feature_dim)
            meta_features = candidate_meta_features[slate_indices]
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
    
    def generate_buffer(self, buffer_size, state_dim, action_dim):
        '''
        @output:
        - buffer: {'observation': {'user_profile': {'user_id': (L,), 
                                                    'uf_{feature_name}': (L, feature_dim)}, 
                                   'user_history': {'history': (L, max_H), 
                                                    'history_if_{feature_name}': (L, max_H * feature_dim), 
                                                    'history_{response}': (L, max_H), 
                                                    'history_length': (L,)}}
                   'policy_output': {'state': (L, state_dim), 'action': (L, action_dim)}, 
                   'next_observation': same format as @output-buffer['observation'], 
                   'reward': (L,),
                   'response': {'done': (L,), 'retention':, (L,)}}
        '''
        observation = self.create_observation_buffer(buffer_size)
        next_observation = self.create_observation_buffer(buffer_size)
        policy_output = {'state': torch.zeros(buffer_size, state_dim).to(torch.float).to(self.device), 
                         'action': torch.zeros(buffer_size, action_dim).to(torch.float).to(self.device)}
        reward = torch.zeros(buffer_size).to(torch.float).to(self.device)
        responses = {'done': torch.zeros(buffer_size).to(torch.bool).to(self.device), 
                     'retention': torch.zeros(buffer_size).to(torch.float).to(self.device)}
        
        return {'observation': observation, 
                'policy_output': policy_output, 
                'reward': reward, 
                'next_observation': next_observation, 
                'user_response': responses}
        
        
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
    
        
        
        
        