import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils
from model.agent.reward_func import *

class BaseOnlineAgent():
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - n_iter
        - train_every_n_step
        - start_train_at_step
        - initial_greedy_epsilon
        - final_greedy_epsilon
        - elbow_greedy
        - check_episode
        - save_episode
        - save_path
        - batch_size
        - actor_lr
        - actor_decay
        '''
        parser.add_argument('--n_iter', type=int, nargs='+', default=[2000], 
                            help='number of training iterations')
        parser.add_argument('--train_every_n_step', type=int, default=1, 
                            help='the number of episode sampling step per training step')
        parser.add_argument('--reward_func', type=str, default='get_immediate_reward', 
                            help='check model.agent.reward_func for reward function name')
        parser.add_argument('--single_response', action='store_true', 
                            help='use only one response type to get reward')
        parser.add_argument('--start_train_at_step', type=int, default=1000,
                            help='start timestamp for buffer sampling')
        parser.add_argument('--initial_greedy_epsilon', type=float, default=0.6, 
                            help='greedy probability for epsilon greedy exploration')
        parser.add_argument('--final_greedy_epsilon', type=float, default=0.05, 
                            help='greedy probability for epsilon greedy exploration')
        parser.add_argument('--elbow_greedy', type=float, default=0.5, 
                            help='greedy probability for epsilon greedy exploration')
        parser.add_argument('--check_episode', type=int, default=100, 
                            help='number of iterations to check output and evaluate')
        parser.add_argument('--test_episode', type=int, default=1000, 
                            help='number of iterations to test')
        parser.add_argument('--save_episode', type=int, default=1000, 
                            help='number of iterations to save current agent')
        parser.add_argument('--save_path', type=str, required=True, 
                            help='save path for networks')
        parser.add_argument('--batch_size', type=int, default=64, 
                            help='training batch size')
        parser.add_argument('--actor_lr', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--actor_decay', type=float, default=1e-4, 
                            help='regularization factor for actor learning')
        parser.add_argument('--explore_rate', type=float, default=1, 
                            help='probability of engaging exploration')
        return parser
    
    def __init__(self, *input_args):
        args, actor, env, buffer = input_args
        
        self.device = args.device
        self.n_iter = [0] + args.n_iter
        self.train_every_n_step = args.train_every_n_step
        self.start_train_at_step = args.start_train_at_step
        self.check_episode = args.check_episode
        self.test_episode = args.test_episode
        self.save_episode = args.save_episode
        self.save_path = args.save_path
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size
        self.actor_lr = args.actor_lr
        self.actor_decay = args.actor_decay
        self.reward_func = eval(args.reward_func)
        self.single_response = args.single_response
        self.explore_rate = args.explore_rate
        
        self.env = env
        if self.single_response:
            self.immediate_response_weight = torch.zeros(len(self.env.response_types)).to(self.device)
            self.immediate_response_weight[0] = 1
        else:
            self.immediate_response_weight = torch.FloatTensor(self.env.response_weights).to(self.device)

        self.exploration_scheduler = utils.LinearScheduler(int(sum(args.n_iter) * args.elbow_greedy), 
                                                           args.final_greedy_epsilon, 
                                                           initial_p=args.initial_greedy_epsilon)
        
        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)
        
        self.buffer = buffer
        
        # create new report file if train the first time
        if len(self.n_iter) == 2:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
            with open(self.save_path + "_test.report", 'w') as outfile:
                outfile.write(f"{args}\n")
                
    
    def train(self):
        # load model parameters if continue training
        if len(self.n_iter) > 2:
            self.load()
        
        t = time.time()
        print("Run procedures before training")
        observation = self.action_before_train()
        t = time.time()
        start_time = t
        
        # training
        print("Training:")
        step_offset = sum(self.n_iter[:-1])
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
            # online episode step
            do_buffer_update = True
            do_explore = True
            observation = self.run_episode_step(i, self.exploration_scheduler.value(i), observation, 
                                                do_buffer_update, do_explore)
            # training step
            if i % self.train_every_n_step == 0:
                self.step_train()
            # log report
            if i % self.check_episode == 0 and i >= self.check_episode:
                t_prime = time.time()
                print(f"Episode step {i}, time diff {t_prime - t}, total time diff {t - start_time})")
                episode_report, train_report = self.get_report()
                log_str = f"step: {i} @ online episode: {episode_report} @ training: {train_report}\n"
                with open(self.save_path + ".report", 'a') as outfile:
                    outfile.write(log_str)
                print(log_str)
                t = t_prime
            # save model and training info
            if i % self.save_episode == 0:
                self.save()
            if i % self.test_episode == 0:
                observation = self.test(i, observation)
                    
        self.action_after_train()
    
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
        for i in tqdm(range(self.start_train_at_step)):
            do_buffer_update = True
            do_explore = np.random.random() < self.explore_rate
            observation = self.run_episode_step(0, initial_epsilon, observation, 
                                                do_buffer_update, do_explore)
        return observation

        
    def run_episode_step(self, *episode_args):
        '''
        Run one step of user-env interaction
        @input:
        - episode_args: (episode_iter, epsilon, observation, do_buffer_update, do_explore)
        @process:
        - policy.explore_action(): observation, candidate items --> policy_output
        - env.step(): policy_output['action'] --> user_feedback, updated_observation
        - reward_func(): user_feedback --> reward
        - buffer.update(observation, policy_output, user_feedback, updated_observation)
        @output:
        - next_observation
        '''
        episode_iter, epsilon, observation, do_buffer_update, do_explore = episode_args
        self.epsilon = epsilon
        with torch.no_grad():
            # wrap observation and candidate items as batch
            observation['batch_size'] = self.episode_batch_size
            candidate_info = self.env.get_candidate_info(observation)
            # sample action
            input_dict = {'observation': observation, 
                          'candidates': candidate_info, 
                          'action_dim': self.env.action_dim,
                          'action': None, 'response': None,
                          'epsilon': epsilon, 'do_explore': do_explore, 'is_train': False}
            policy_output = self.actor(input_dict)
            # apply action on environment
            # Note: action must be indices on env.candidate_iids
            action_dict = {'action': policy_output['action']}   
            new_observation, user_feedback, updated_observation = self.env.step(action_dict)
            # calculate reward
            user_feedback['immediate_response_weight'] = self.immediate_response_weight
            R = self.reward_func(user_feedback).detach()
            user_feedback['reward'] = R
            self.eval_history['avg_reward'].append(R.mean().item())
#             self.eval_history['min_reward'].append(R.min().cpu().numpy())
            self.eval_history['max_reward'].append(R.max().item())
            self.eval_history['reward_variance'].append(torch.var(R).item())
            self.eval_history['coverage'].append(user_feedback['coverage'])
            self.eval_history['intra_slate_diversity'].append(user_feedback['ILD'])
            for i,resp in enumerate(self.env.response_types):
                self.eval_history[f'{resp}_rate'].append(user_feedback['immediate_response'][:,:,i].mean().item())
#             mean_response = torch.mean(user_feedback, dim = 0)
#             for i,f in enumerate(self.env.response_types):
#                 self.eval_history[f].append(user_feedback[])
            # update replay buffer
            if do_buffer_update:
                self.buffer.update(observation, policy_output, user_feedback, updated_observation)
        return new_observation
    
    def initialize_training_history(self):
        '''
        Specify the metrics (e.g. training_loss) to observe in self.training_history 
        '''
        self.training_history = {k: [] for k in self.actor.get_loss_observation()}
    
    def get_report(self):
        '''
        get report dict to write
        '''
        episode_report = {k: np.mean(v[-self.check_episode:]) for k,v in self.eval_history.items()}
#         episode_report.update(self.env.get_env_report())
        train_report = {k: np.mean(v[-self.check_episode:]) for k,v in self.training_history.items()}
        return episode_report, train_report
    
    def action_after_train(self):
        self.env.stop()
        
    ###############################
    #   Requires implementation   #
    ###############################
    
    
    def step_train(self):
        '''
        @process:
        - buffer.sample(): batch_size --> observation, policy_output, user_response
            - observation:{
                'user_profile':{
                    'user_id': (B,)
                    'uf_{feature_name}': (B,feature_dim), the user features}
                'user_history':{
                    'history': (B,max_H)
                    'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}}
            - target_output: {
                'state': (B,state_dim), 
                'prob': (B,L),
                'action': (B,K),
                'reg': scalar}
            - user_response: {
                'reward': (B,),
                'immediate_response': (B,K*n_feedback)}
        - policy.get_forward(): observation, candidates --> policy_output
        - policy.get_loss(): observation, candidates, policy_output, user_response --> loss
        - optimizer.zero_grad(); loss.backward(); optimizer.step()
        - update training history
        '''
        observation, target_output, target_response, _, __ = self.buffer.sample(self.batch_size)
        
        # forward pass
        observation['batch_size'] = self.episode_batch_size
        candidate_info = self.env.get_candidate_info(observation)
        input_dict = {'observation': observation, 
                      'candidates': candidate_info, 
                      'action_dim': self.env.action_dim,
                      'action': target_output['action'], 
                      'response': target_response,
                      'epsilon': 0, 'do_explore': False, 'is_train': True}
        policy_output = self.actor(input_dict)

        # loss
        policy_output['action'] = target_output['action']
        policy_output.update(target_response)
        policy_output.update({'immediate_response_weight': self.immediate_response_weight})

        loss_dict = self.actor.get_loss(input_dict, policy_output)
        actor_loss = loss_dict['loss']
        # optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for k in loss_dict:
            try:
                self.training_history[k].append(loss_dict[k].item())
            except:
                self.training_history[k].append(loss_dict[k])
#         self.training_history['actor_loss'].append(actor_loss.item())


        return {"step_loss": (self.training_history['loss'][-1])}
    
    def test(self, *episode_args):
        '''
        Run one step of user-env interaction
        @input:
        - episode_args: (episode_iter, epsilon, observation, do_buffer_update, do_explore)
        @process:
        - policy.explore_action(): observation, candidate items --> policy_output
        - env.step(): policy_output['action'] --> user_feedback, updated_observation
        - reward_func(): user_feedback --> reward
        - buffer.update(observation, policy_output, user_feedback, updated_observation)
        @output:
        - next_observation
        '''
        episode_iter, observation = episode_args
        test_report = {}
        with torch.no_grad():
            # wrap observation and candidate items as batch
            observation['batch_size'] = self.episode_batch_size
            candidate_info = self.env.get_candidate_info(observation)
            # sample action
            input_dict = {'observation': observation, 
                          'candidates': candidate_info, 
                          'action_dim': self.env.action_dim,
                          'action': None, 'response': None,
                          'epsilon': 0, 'do_explore': False, 'is_train': False}
            policy_output = self.actor(input_dict)
            # apply action on environment
            # Note: action must be indices on env.candidate_iids
            # Note: action in policy_output are indices on the selected candidate_info, but candidate_info is always the entire set, so it can be directly used as action on the environment.
            action_dict = {'action': policy_output['action']}   
            new_observation, user_feedback, updated_observation = self.env.step(action_dict)
            # calculate reward
            user_feedback['immediate_response_weight'] = self.immediate_response_weight
            R = self.reward_func(user_feedback).detach()
            user_feedback['reward'] = R
            test_report['avg_reward'] = R.mean().item()
#             self.eval_history['min_reward'].append(R.min().cpu().numpy())
            test_report['max_reward'] = R.max().item()
            test_report['reward_variance'] = torch.var(R).item()
            test_report['coverage'] = user_feedback['coverage']
            test_report['intra_slate_diversity'] = user_feedback['ILD']
            for j,resp in enumerate(self.env.response_types):
                test_report[f'{resp}_rate'] = user_feedback['immediate_response'][:,:,j].mean().item()
        train_report = {k: np.mean(v[-self.check_episode:]) for k,v in self.training_history.items()}
        log_str = f"step: {episode_iter} @ online episode: {test_report} @ training: {train_report}\n"
        with open(self.save_path + "_test.report", 'a') as outfile:
            outfile.write(log_str)
        return new_observation
    

    def save(self):
        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")


    def load(self):
        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
        
        