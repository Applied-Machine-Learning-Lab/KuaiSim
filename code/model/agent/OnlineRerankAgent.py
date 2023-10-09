import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils
from model.agent.reward_func import *
from model.agent.BaseOnlineAgent import BaseOnlineAgent

class OnlineRerankAgent(BaseOnlineAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - from BaseOnlineAgent:
            - n_iter
            - train_every_n_step
            - start_train_at_step
            - reward_func
            - single_response
            - initial_greedy_epsilon
            - final_greedy_epsilon
            - elbow_greedy
            - check_episode
            - test_episode
            - save_episode
            - save_path
            - batch_size
            - actor_lr
            - actor_decay
            - explore_rate
        '''
        parser = BaseOnlineAgent.parse_model_args(parser)
        parser.add_argument('--learn_initial_during_rerank', action='store_true', 
                            help='learning initial ranker when learning reranker')
        return parser
    
    def __init__(self, *input_args):
        # env, actor, buffer, device
        # # n_iter, train_every_n_step, start_train_at_step, 
        # check_episode, test_episode, save_episode, save_path, 
        # episode_batch_size, batch_size, actor_lr, actor_decay, 
        # reward_func, single_response, explore_rate
        args, actor, env, buffer = input_args
        self.learn_initial_during_rerank = args.learn_initial_during_rerank
        super().__init__(args, actor, env, buffer)
        
        
    def action_before_train(self):
        '''
        Action before training:
        - facade setup:
            - buffer setup
        - run random episodes to build-up the initial buffer
        '''
        # buffer setup
        self.buffer.reset(self.env, self.actor)
        
        # training records
        self.training_history = {}
        self.eval_history = {'avg_reward': [], 'max_reward': [], 'reward_variance': [], 
                             'coverage': [], 'intra_slate_diversity': []}
        self.eval_history.update({f'{resp}_rate': [] for resp in self.env.response_types})
#         self.test_history = {k:[] for k in self.eval_history}
        self.initialize_training_history()
        
        # random explore before training
        initial_epsilon = 1.0
        observation = self.env.reset()
        self.actor.train_initial = True
        self.actor.train_rerank = False
        for i in tqdm(range(self.start_train_at_step)):
            do_buffer_update = True
            do_explore = np.random.random() < self.explore_rate
            observation = self.run_episode_step(0, initial_epsilon, observation, 
                                                do_buffer_update, do_explore)
            if i % self.train_every_n_step == 0:
                self.step_train()
        self.actor.train_initial = self.learn_initial_during_rerank
        self.actor.train_rerank = True
        return observation