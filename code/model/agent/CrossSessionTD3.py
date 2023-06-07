import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.BaseRLAgent import BaseRLAgent
    
class CrossSessionTD3(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_lr
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
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
        parser.add_argument('--critic_lr', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
                            help='mitigation factor')
        parser.add_argument('--noise_std', type=float, default=0.1, 
                            help='noise standard deviation for action exploration')
        parser.add_argument('--noise_clip', type=float, default=1.0, 
                            help='noise clip bound for action exploration')
        return parser
    
    def __init__(self, *input_args):
        '''
        
        from BaseRLAgent:
        - all args
        - self.device
        - self.env
        - self.actor
        - self.buffer
        '''
        args, env, actor, critics, buffer = input_args
        super().__init__(args, env, actor, buffer)
        
        self.noise_std = args.noise_std
        self.noise_clip = args.noise_clip
        
        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.tau = args.target_mitigate_coef 
        
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)

        self.critic1 = critics[0]
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)
        
        self.critic2 = critics[1]
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)

        # register models that will be saved
        self.registered_models.append((self.critic1, self.critic1_optimizer, "_critic1"))
        self.registered_models.append((self.critic2, self.critic2_optimizer, "_critic2"))
        
        
        
    def setup_monitors(self):
        '''
        This is used in super().action_before_train() in super().train()
        Then the agent will call several rounds of run_episode_step() for collecting initial random data samples
        '''
        super().setup_monitors()
        self.training_history.update({'Q1_loss': [], 'Q2_loss': [],
                                      'Q1': [], 'Q2': [], 'next_Q1': [], 'next_Q2': [], 'target_Q': []})
        self.eval_history.update({'avg_retention': [], 'max_retention': [], 'min_retention': []})
                                     
                                     
    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update, do_explore = episode_args
        self.epsilon = epsilon
        is_train = False
        with torch.no_grad():
            # sample action
            policy_output = self.apply_policy(observation, self.actor, 
                                              epsilon, do_explore, is_train)
            # apply action on environment and update replay buffer
            action_dict = {'action': policy_output['action']}
            new_observation, user_feedback, updated_info = self.env.step(action_dict)
            # calculate reward
            R = self.get_reward(user_feedback)
            user_feedback['reward'] = R
            self.current_sum_reward = self.current_sum_reward + R
            done_mask = user_feedback['done']
                                     
            # monitor update
            if torch.sum(done_mask) > 0:
                self.eval_history['avg_retention'].append(user_feedback['retention'].mean().item()) 
                self.eval_history['max_retention'].append(user_feedback['retention'].max().item()) 
                self.eval_history['min_retention'].append(user_feedback['retention'].min().item())
                self.eval_history['avg_total_reward'].append(self.current_sum_reward.mean().item())
                self.eval_history['max_total_reward'].append(self.current_sum_reward.max().item())
                self.eval_history['min_total_reward'].append(self.current_sum_reward.min().item())
                self.current_sum_reward *= 0
            
            self.eval_history['avg_reward'].append(R.mean().item())
            self.eval_history['reward_variance'].append(torch.var(R).item())
            for i,resp in enumerate(self.env.response_types):
                self.eval_history[f'{resp}_rate'].append(user_feedback['immediate_response'][:,:,i].mean().item())
            # update replay buffer
            if do_buffer_update:
                self.buffer.update(observation, policy_output, user_feedback, updated_info['updated_observation'])
            observation = new_observation
        return new_observation
            

    def step_train(self):
        '''
        @process:
        '''
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)
        reward = user_feedback['reward']
        reward = reward.to(torch.float)
        done_mask = done_mask.to(torch.float)
        
        critic_loss_list, actor_loss = self.get_td3_loss(observation, policy_output, reward, done_mask, next_observation)
        target_Q, next_Q1, next_Q2, Q1_loss, Q1, Q2_loss, Q2 = critic_loss_list
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['Q1_loss'].append(Q1_loss)
        self.training_history['Q2_loss'].append(Q2_loss)
        self.training_history['Q1'].append(Q1)
        self.training_history['Q2'].append(Q2)
        self.training_history['next_Q1'].append(next_Q1)
        self.training_history['next_Q2'].append(next_Q2)
        self.training_history['target_Q'].append(target_Q)

        # Update the frozen target models
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    
    def get_td3_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                     do_actor_update = True, do_critic_update = True):
        '''
        @input:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H, feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
        - policy_output: {'state': (B, state_dim), 
                          'action: (B, action_dim)}
        - reward: (B,)
        - done_mask: (B,)
        - next_observation: the same format as @input-observation
        '''
        is_train = True
        
        # Compute the target Q value
        next_policy_output = self.apply_policy(next_observation, self.actor_target, self.epsilon, True, is_train)
        target_critic1_output = self.apply_critic(next_observation, next_policy_output, self.critic1_target)
        target_critic2_output = self.apply_critic(next_observation, next_policy_output, self.critic2_target)
        next_Q1, next_Q2 = target_critic1_output['q'], target_critic2_output['q']
        # TD3 target reward
        target_Q = reward + self.gamma *  (1 - done_mask) * torch.min(next_Q1, next_Q2).detach() 
#         # RLUR target reward: r+gamma*Q' when done; r+Q when not done
#         target_Q = reward + ((self.gamma * done_mask) + (1 - done_mask)) * target_Q.detach() 
#         # bandit
#         target_Q = reward 

        # [target Q, next_Q1, next_Q2, Q1 TD loss, Q1, Q2 TD loss, Q2]
        critic_loss_list = [target_Q.mean().item(), next_Q1.mean().item(), next_Q2.mean().item()]
        if do_critic_update and self.critic_lr > 0:
            for critic, optimizer in [(self.critic1, self.critic1_optimizer), 
                                           (self.critic2, self.critic2_optimizer)]:
                # Get current Q estimate
                current_critic_output = self.apply_critic(observation, policy_output, critic)
                current_Q = current_critic_output['q']
                # Compute critic loss
                critic_loss = F.mse_loss(current_Q, target_Q).mean()
                critic_loss_list.append(critic_loss.item())
                critic_loss_list.append(torch.mean(current_Q).item())

                # Optimize the critic
                optimizer.zero_grad()
                critic_loss.backward()
                optimizer.step()

        # Compute actor loss
        temp_policy_output = self.apply_policy(observation, self.actor, 
                                               self.epsilon, self.do_explore_in_train, is_train)
        critic_output = self.apply_critic(observation, temp_policy_output, self.critic1)
        actor_loss = -critic_output['q'].mean()

        if do_actor_update and self.actor_lr > 0:
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
        return critic_loss_list, actor_loss


    def apply_policy(self, observation, actor, *policy_args):
        '''
        @input:
        - observation:{'user_profile':{
                           'user_id': (B,)
                           'uf_{feature_name}': (B,feature_dim), the user features}
                       'user_history':{
                           'history': (B,max_H)
                           'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
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
        out_dict = self.actor(observation)
        
        if do_explore:
            action = out_dict['action']
            # sampling noise of action embedding
            if np.random.rand() < epsilon:
                action = torch.clamp(torch.rand_like(action)*self.noise_std, -self.noise_clip, self.noise_clip)
            else:
                action = action + torch.clamp(torch.rand_like(action)*self.noise_std, 
                                                      -self.noise_clip, self.noise_clip)
            out_dict['action'] = action
        return out_dict
    
    def apply_critic(self, observation, policy_output, critic):
        feed_dict = {'state': policy_output['state'],
                     'action': policy_output['action']}
        return critic(feed_dict)