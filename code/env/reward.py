import torch

def sum_with_cost(feedback, zero_reward_cost = 0.1):
    '''
    @input:
    - feedback: (B, L)
    @output:
    - reward: (B,)
    '''
    B,L = feedback.shape
    cost = torch.zeros_like(feedback)
    cost[feedback == 0] = -zero_reward_cost
    reward = torch.sum(feedback + cost, dim = -1)
    return reward


def sigmoid_sum_with_cost(feedback, zero_reward_cost = 0.1):
    '''
    @input:
    - feedback: (B, L)
    @output:
    - reward: (B,)
    '''
    reward = sum_with_cost(feedback, zero_reward_cost)
    return torch.sigmoid(reward)


def log_sum_with_cost(feedback, zero_reward_cost = 0.1):
    '''
    @input:
    - feedback: (B, L)
    @output:
    - reward: (B,)
    '''
    reward = sum_with_cost(feedback, zero_reward_cost)
    reward[reward>0] = (reward[reward>0]+1).log()
    return torch.sigmoid(reward)

def mean_with_cost(feedback, zero_reward_cost = 0.1):
    '''
    @input:
    - feedback: (B, L)
    @output:
    - reward: (B,)
    '''
    B,L = feedback.shape
    cost = torch.zeros_like(feedback)
    cost[feedback == 0] = -zero_reward_cost
    reward = torch.mean(feedback + cost, dim = -1)
    return reward

def mean_advance_with_cost(feedback, zero_reward_cost = 0.1, offset = 0.5):
    '''
    @input:
    - feedback: (B, L)
    @output:
    - reward: (B,)
    '''
    B,L = feedback.shape
    cost = torch.zeros_like(feedback)
    cost[feedback == 0] = -zero_reward_cost
    reward = torch.mean(feedback + cost, dim = -1) - offset
    return reward