import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

import utils
from model.reward import *
from model.agent.BaseRLAgent import BaseRLAgent

class ListRecOnlineAgent(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - test_episode
        - from BaseRLAgent:
            - gamma
            - reward_func
            - n_iter
            - train_every_n_step
            - start_policy_train_at_step
            - initial_epsilon
            - final_epsilon
            - elbow_epsilon
            - explore_rate
            - do_explore_in_train
            - check_episode
            - save_episode
            - save_path
            - actor_lr
            - actor_decay
            - batch_size
        '''
        parser = BaseRLAgent.parse_model_args(parser)
        # monitoring
        parser.add_argument('--test_episode', type=int, default=1000, 
                            help='number of iterations to do ranking test')
        return parser
    
    def __init__(self, *input_args):
        '''
        device
        n_iter, gamma, reward_func
        train_every_n_step, start_policy_train_at_step
        initial_epsilon, final_epsilon, elbow_epsilon, explore_rate, do_explore_in_train
        check_episode, save_episode, save_path, test_episode
        batch_size, actor_lr, actor_decay
        env, actor, buffer
        exploration_scheduler, actor_optimizer
        registered_models
        '''
        
        args, env, actor, buffer = input_args
        
        super().__init__(*input_args)
        
        self.test_episode = args.test_episode
        
        # create new report file if train the first time
        if len(self.n_iter) == 2:
            with open(self.save_path + "_test.report", 'w') as outfile:
                outfile.write(f"{args}\n")
                
    def setup_monitors(self):
        '''
        This is used in super().action_before_train() in super().train()
        Then the agent will call several rounds of run_episode_step() for collecting initial random data samples
        '''
        super().setup_monitors()
        self.training_history.update({loss_key: [] for loss_key in self.actor.get_loss_observation()})
    
    def train(self):
        # load model parameters if continue training
        if len(self.n_iter) > 2:
            self.load()
        
        t = time.time()
        print("Run procedures before training")
        self.action_before_train()
        t = time.time()
        start_time = t
        
        # training
        print("Training:")
        step_offset = sum(self.n_iter[:-1])
        do_buffer_update = True
        observation = deepcopy(self.env.current_observation)
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
            do_explore = True # always do online exploration
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
                self.test(i, observation)
                    
        self.action_after_train()
        
       
    def apply_policy(self, observation, actor, *policy_args):
        '''
        @input:
        - observation:{'user_profile':{
                           'user_id': (B,)
                           'uf_{feature_name}': (B,feature_dim), the user features}
                       'user_history':{
                           'history': (B,max_H)
                           'history_if_{feature_name}': (B,max_H,feature_dim), the history item features, 
                       'action': (B, slate_size),  # or None during inference
                       'response': (B, slate_size, n_feedback)   # or None during inference}
        - actor: the actor model
        - epsilon: scalar
        - do_explore: boolean
        - is_train: boolean
        @output:
        - policy_output
        '''
        epsilon = policy_args[0]
        do_explore = policy_args[1]
        is_train = policy_args[2]
        input_dict = {'observation': observation, 
                      'candidates': self.env.get_candidate_info(observation), 
                      'action_dim': self.env.slate_size,
                      'action': observation['action'] if 'action' in observation else None,  
                      'response': observation['response'] if 'response' in observation else None, # None during inference
                      'epsilon': epsilon, 
                      'do_explore': do_explore, 
                      'is_train': is_train, 
                      'batch_wise': False}
        out_dict = self.actor(input_dict)
        return out_dict

        
    ###############################
    #   Requires implementation   #
    ###############################
    
    
    def step_train(self):
        '''
        @process:
        - buffer.sample(): batch_size --> observation, target_output, target_response
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
            - target_response: {
                'reward': (B,),
                'immediate_response': (B,K*n_feedback)}
        - policy.get_forward(): observation, candidates --> policy_output
        - policy.get_loss(): observation, candidates, policy_output, user_response --> loss
        - optimizer.zero_grad(); loss.backward(); optimizer.step()
        - update training history
        '''
        observation, target_output, target_response, done_mask, _ = self.buffer.sample(self.batch_size)
        
        # forward pass
        observation.update({'action': target_output['action'], 'response': target_response})
        policy_output = self.apply_policy(observation, self.actor, 0, False, True)

        # loss
        policy_output['action'] = target_output['action']
        policy_output.update(target_response)
        policy_output.update({'immediate_response_weight': self.env.response_weights})

        loss_dict = self.actor.get_loss(observation, policy_output)
        actor_loss = loss_dict['loss']
        if 'actor_loss' not in loss_dict:
            loss_dict['actor_loss'] = actor_loss
        
        # optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for k in loss_dict:
            try:
                self.training_history[k].append(loss_dict[k].item())
            except:
                self.training_history[k].append(loss_dict[k])

        return loss_dict

    
    def test(self, *episode_args):
        '''
        Run one step of user-env interaction with greedy strategy
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
            # no exploration during test inference
            do_explore = False
            policy_output = self.apply_policy(observation, self.actor, 0, do_explore, False)
            
            # apply action on environment
            # Note: action must be indices on env.candidate_iids
            # Note: action in policy_output are indices on the selected candidate_info, but candidate_info is always the entire set, so it can be directly used as action on the environment.
            action_dict = {'action': policy_output['action']}   
            _, response_dict, updated_observation = self.env.step(action_dict, update_observation = False)
            
            # calculate reward
            R = self.get_reward(response_dict)
            
            response_dict['reward'] = R
            test_report['avg_reward'] = R.mean().item()
            test_report['min_reward'] = R.min().item()
            test_report['max_reward'] = R.max().item()
            test_report['reward_variance'] = torch.var(R).item()
            test_report['coverage'] = response_dict['coverage']
            test_report['ILD'] = response_dict['ILD']
            for j,resp in enumerate(self.env.response_types):
                test_report[f'{resp}_rate'] = response_dict['immediate_response'][:,:,j].mean().item()
        train_report = {k: np.mean(v[-self.check_episode:]) for k,v in self.training_history.items()}
        log_str = f"step: {episode_iter} @ online episode: {test_report} @ training: {train_report}\n"
        with open(self.save_path + "_test.report", 'a') as outfile:
            outfile.write(log_str)
        return updated_observation
    

        