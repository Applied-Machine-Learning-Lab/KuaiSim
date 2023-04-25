import torch
import torch.nn.functional as F
import random
import numpy as np

import utils
from model.buffer.BaseBuffer import BaseBuffer

class HyperActorBuffer(BaseBuffer):
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - from BaseBuffer:
            - buffer_size
        '''
        parser = BaseBuffer.parse_model_args(parser)
        return parser
        
    def reset(self, *reset_args):
        
        '''
        @output:
        - buffer: {'observation': {'user_profile': {'user_id': (L,), 
                                                    'uf_{feature_name}': (L, feature_dim)}, 
                                   'user_history': {'history': (L, max_H), 
                                                    'history_if_{feature_name}': (L, max_H * feature_dim), 
                                                    'history_{response}': (L, max_H), 
                                                    'history_length': (L,)}}
                   'policy_output': {'state': (L, state_dim), 
                                     'action': (L, action_dim) = (L, slate_size), 
                                     'hyper_action': (L, hyper_action_size)},
                   'next_observation': same format as @output-buffer['observation'], 
                   'done_mask': (L,),
                   'response': {'reward': (L,), 
                                'immediate_response':, (L, action_dim * response_dim)}}
        '''
        env = reset_args[0]
        actor = reset_args[1]
        
        super().reset(env, actor)
        
        self.buffer['user_response']['immediate_response'] = torch.zeros(self.buffer_size, 
                                                                         env.response_dim * actor.effect_action_dim)\
                                                         .to(torch.float).to(self.device)
        self.buffer['policy_output']['action'] = torch.zeros(self.buffer_size, actor.hyper_action_dim)\
                                                         .to(torch.float).to(self.device)
        self.buffer['policy_output']['effect_action'] = torch.zeros(self.buffer_size, actor.effect_action_dim)\
                                                         .to(torch.long).to(self.device)
        
        return self.buffer
    
    
    
    def sample(self, batch_size):
        '''
        Batch sample is organized as a tuple of (observation, policy_output, user_response, done_mask, next_observation)
        
        Buffer: see reset@output
        '''
        # get indices
        indices = np.random.randint(0, self.current_buffer_size, size = batch_size)
        # observation
        profile = {k:v[indices] for k,v in self.buffer["observation"]["user_profile"].items()}
        history = {k:v[indices] for k,v in self.buffer["observation"]["user_history"].items()}
        observation = {"user_profile": profile, "user_history": history}
        # next observation
        profile = {k:v[indices] for k,v in self.buffer["next_observation"]["user_profile"].items()}
        history = {k:v[indices] for k,v in self.buffer["next_observation"]["user_history"].items()}
        next_observation = {"user_profile": profile, "user_history": history}
        # policy output
        policy_output = {"state": self.buffer["policy_output"]["state"][indices], 
                         "action": self.buffer["policy_output"]["action"][indices], 
                         "hyper_action": self.buffer["policy_output"]["action"][indices], 
                         "effect_action": self.buffer["policy_output"]["effect_action"][indices]}  # main change to BaseBuffer
        # user response
        user_response = {"reward": self.buffer["user_response"]["reward"][indices], 
                         "immediate_response": self.buffer["user_response"]["immediate_response"][indices]}
        # done mask
        done_mask = self.buffer["done_mask"][indices]
        return observation, policy_output, user_response, done_mask, next_observation
    
    def update(self, observation, policy_output, user_feedback, next_observation):
        '''
        @input:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H * feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B,)}}
        - policy_output: {'user_state': (B, state_dim), 
                          'prob': (B, action_dim),
                          'action': (B, action_dim),
                          'hyper_action': (B, hyper_action_dim)}
        - user_feedback: {'done': (B,), 
                          'immdiate_response':, (B, action_dim * feedback_dim), 
                          'reward': (B,)}
        - next_observation: same format as update_buffer@input-observation
        '''
        # get buffer indices to update
        B = len(user_feedback['reward'])
        if self.buffer_head + B >= self.buffer_size:
            tail = self.buffer_size - self.buffer_head
            indices = [self.buffer_head + i for i in range(tail)] + \
                        [i for i in range(B - tail)]
        else:
            indices = [self.buffer_head + i for i in range(B)]
        indices = torch.tensor(indices).to(torch.long).to(self.device)
        
        # update buffer - observation
        for k,v in observation['user_profile'].items():
            self.buffer['observation']['user_profile'][k][indices] = v
        for k,v in observation['user_history'].items():
            self.buffer['observation']['user_history'][k][indices] = v
        # update buffer - next observation
        for k,v in next_observation['user_profile'].items():
            self.buffer['next_observation']['user_profile'][k][indices] = v
        for k,v in next_observation['user_history'].items():
            self.buffer['next_observation']['user_history'][k][indices] = v
        # update buffer - policy output
        self.buffer['policy_output']['state'][indices] = policy_output['state']
        self.buffer['policy_output']['action'][indices] = policy_output['hyper_action']
        self.buffer['policy_output']['effect_action'][indices] = policy_output['effect_action'] # main change to BaseBuffer
        # update buffer - user response
        self.buffer['user_response']['immediate_response'][indices] = user_feedback['immediate_response'].view(B,-1)
        self.buffer['user_response']['reward'][indices] = user_feedback['reward']
        # update buffer - done
        self.buffer['done_mask'][indices] = user_feedback['done']
        
        # buffer pointer
        self.buffer_head = (self.buffer_head + B) % self.buffer_size
        self.n_stream_record += B
        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)
        
        