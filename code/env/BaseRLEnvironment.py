from argparse import Namespace
import torch

import utils
from model.simulator import *
from reader import *

class BaseRLEnvironment():
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - max_step_per_episode
        - initial_temper
        '''
        parser.add_argument('--max_step_per_episode', type=int, default=100, help='max number of iteration allowed in each episode')
        parser.add_argument('--initial_temper', type=float, default=10, help='initial temper of users')
        return parser
    
    
    def __init__(self, args):
        self.device = args.device
        super().__init__()
        self.max_step_per_episode = args.max_step_per_episode
        self.initial_temper = args.initial_temper
        
    def reset(self, params):
        pass
        
    def step(self, action):
        pass
    
    
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
        training_args.device = self.device
        infile.close()
        readerClass = eval('{0}.{0}'.format(class_args.reader))
        reader = readerClass(training_args)
        return reader, training_args
    
    
    def get_observation_from_batch(self, sample_batch):
        '''
        extract observation from the reader's sample batch
        @input:
        - sample_batch: {
            'user_id': (B,)
            'uf_{feature}': (B,F_dim(feature)), user features
            'history': (B,max_H)
            'history_length': (B,)
            'history_if_{feature}': (B, max_H * F_dim(feature))
            'history_{response}': (B, max_H)
            ... user ground truth feedbacks are not included as observation
        }
        @output:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H, feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
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
    