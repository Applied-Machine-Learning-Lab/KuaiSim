import torch

def get_retention_reward(user_feedback, reward_base = 0.7):
    '''
    @input:
    - user_feedback: {'retention': (B,), ...}
    @output:
    - reward: (B,)
    '''
    reward = - user_feedback['retention']/10.0
    return reward

def get_immediate_reward(user_feedback):
    '''
    @input:
    - user_feedback: {'immediate_response': (B, slate_size, n_feedback), 
                      'immediate_response_weight': (n_feedback),
                      ... other feedbacks}
    @output:
    - reward: (B,)
    '''
    # (B, slate_size, n_feedback)
    if 'immediate_response_weight' in user_feedback:
        point_reward = user_feedback['immediate_response'] * user_feedback['immediate_response_weight'].view(1,1,-1)
    else:
        point_reward = user_feedback['immediate_response']
    # (B, slate_size)
    combined_reward = torch.sum(point_reward, dim = 2)
    # (B,)
    #leave_reward = user_feedback['leave'] * user_feedback['leave_weight']
    # (B,)
    #reward = point_reward.sum(dim = -1) + leave_reward
    reward = torch.mean(combined_reward, dim = 1)
    return reward

def get_immediate_reward_sum(user_feedback):
    '''
    @input:
    - user_feedback: {'immediate_response': (B, slate_size, n_feedback), 
                      'immediate_response_weight': (n_feedback),
                      ... other feedbacks}
    @output:
    - reward: (B,)
    '''
    # (B, slate_size, n_feedback)
    if 'immediate_response_weight' in user_feedback:
        point_reward = user_feedback['immediate_response'] * user_feedback['immediate_response_weight'].view(1,1,-1)
    else:
        point_reward = user_feedback['immediate_response']
    # (B, slate_size)
    combined_reward = torch.sum(point_reward, dim = 2)
    # (B,)
    #leave_reward = user_feedback['leave'] * user_feedback['leave_weight']
    # (B,)
    #reward = point_reward.sum(dim = -1) + leave_reward
    reward = torch.sum(combined_reward, dim = 1)
    return reward
    

def sum_with_cost(feedback, zero_reward_cost = 0.1):
    '''
    @input:
    - feedback: (B, K)
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
    - feedback: (B, K)
    @output:
    - reward: (B,)
    '''
    reward = sum_with_cost(feedback, zero_reward_cost)
    return torch.sigmoid(reward)


def log_sum_with_cost(feedback, zero_reward_cost = 0.1):
    '''
    @input:
    - feedback: (B, K)
    @output:
    - reward: (B,)
    '''
    reward = sum_with_cost(feedback, zero_reward_cost)
    reward[reward>0] = (reward[reward>0]+1).log()
    return torch.sigmoid(reward)

def mean_with_cost(feedback_dict, zero_reward_cost = 0.1):
    '''
    @input:
    - feedback: (B, K)
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
    - feedback: (B, K)
    @output:
    - reward: (B,)
    '''
    B,L = feedback.shape
    cost = torch.zeros_like(feedback)
    cost[feedback == 0] = -zero_reward_cost
    reward = torch.mean(feedback + cost, dim = -1) - offset
    return reward