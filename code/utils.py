from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import os

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def check_folder_exist(fpath):
    if os.path.exists(fpath):
        print("dir \"" + fpath + "\" existed")
    else:
        try:
            os.mkdir(fpath)
        except:
            print("error when creating \"" + fpath + "\"") 
            
def setup_path(fpath, is_dir = True):
    dirs = [p for p in fpath.split("/")]
    curP = ""
    dirs = dirs[:-1] if not is_dir else dirs
    for p in dirs:
        curP += p
        check_folder_exist(curP)
        curP += "/"

###########################################
#              Data Related               #
###########################################

def repeat_n_core(df, user_col_id, item_col_id, n_core, user_counts, item_counts):
    '''
    Iterative n_core filter
    
    @input:
    - df: [UserID, ItemID, ...]
    - n_core: number of core
    - user_counts: {uid: frequency}
    - item_counts: {iid: frequency}
    '''
    print("N-core is set to [5,100]")
    n_core = min(max(n_core, 5),100) # 5 <= n_core <= 100
    print("Filtering " + str(n_core) + "-core data")
    iteration = 0
    lastNRemove = len(df)  # the number of removed record
    proposedData = df.values
    originalSize = len(df)
    
    # each iteration, count number of records that need to delete
    while lastNRemove != 0:
        iteration += 1
        print("Iteration " + str(iteration))
        changeNum = 0
        newData = []
        for row in tqdm(proposedData):
            user, item = row[user_col_id], row[item_col_id]
            if user_counts[user] < n_core or item_counts[item] < n_core:
                user_counts[user] -= 1
                item_counts[item] -= 1
                changeNum += 1
            else:
                newData.append(row)
        proposedData = newData
        print("Number of removed record: " + str(changeNum))
        if changeNum > lastNRemove + 10000:
            print("Not converging, will use original data")
            break
        else:
            lastNRemove = changeNum
    print("Size change: " + str(originalSize) + " --> " + str(len(proposedData)))
    return pd.DataFrame(proposedData, columns=df.columns)
    
def run_multicore(df, user_key = "user_id", item_key = "item_id", n_core = 10, auto_core = False, filter_rate = 0.2):
    '''
    @input:
    - df: pd.DataFrame, col:[UserID,ItemID,...]
    - n_core: number of core
    - auto_core: automatically find n_core, set to True will ignore n_core
    - filter_rate: proportion of removal for user/item, require auto_core = True
    '''
    print(f"Filter {n_core if not auto_core else 'auto'}-core data.")
    uCounts = df[user_key].value_counts().to_dict() # {user_id: count}
    iCounts = df[item_key].value_counts().to_dict() # {item_id: count}
            
    # automatically find n_core based on filter rate
    if auto_core:
        print("Automatically find n_core that filter " + str(100*filter_rate) + "% of user/item")
        
        nCoreCounts = dict() # {n_core: [#user, #item]}
        for v,c in iCounts.items():
            if c not in nCoreCounts:
                nCoreCounts[c] = [0,1]
            else:
                nCoreCounts[c][1] += 1
        for u,c in uCounts.items():
            if c not in nCoreCounts:
                nCoreCounts[c] = [1,0]
            else:
                nCoreCounts[c][0] += 1
                
        # find n_core for: filtered data < filter_rate * length(data)
        userToRemove = 0 # number of user records to remove
        itemToRemove = 0 # number of item records to remove
        for c,counts in sorted(nCoreCounts.items()):
            userToRemove += counts[0] * c # #user * #core
            itemToRemove += counts[1] * c # #item * #core
            if userToRemove > filter_rate * len(df) or itemToRemove > filter_rate * len(df):
                n_core = c
                print("Autocore = " + str(n_core))
                break
    else:
        print("n_core = " + str(n_core))
            
    return repeat_n_core(df, 0, 1, n_core, uCounts, iCounts)

def padding_and_clip(sequence, max_len, padding_direction = 'left'):
    if len(sequence) < max_len:
        sequence = [0] * (max_len - len(sequence)) + sequence if padding_direction == 'left' else sequence + [0] * (max_len - len(sequence))
    sequence = sequence[-max_len:] if padding_direction == 'left' else sequence[:max_len]
    return sequence
    from tqdm import tqdm
import numpy as np

def get_onehot_vocab(meta_df, features):
    print('build vocab for onehot features')
    vocab = {}
    for f in tqdm(features):
        value_list = list(meta_df[f].unique())
        vocab[f] = {}
        for i,v in enumerate(value_list):
            onehot_vec = np.zeros(len(value_list))
            onehot_vec[i] = 1
            vocab[f][v] = onehot_vec
    return vocab

def get_multihot_vocab(meta_df, features, sep = ','):
    print('build vocab for multihot features:')
    vocab = {}
    for f in features:
        print(f'\t{f}')
        ID_freq = {}
        for row in tqdm(meta_df[f]):
            IDs = str(row).split(sep)
            for ID in IDs:
                if ID not in ID_freq:
                    ID_freq[ID] = 1
                else:
                    ID_freq[ID] += 1
        v_list = list(ID_freq.keys())
        vocab[f] = {}
        for i,v in enumerate(v_list):
            onehot_vec = np.zeros(len(v_list))
            onehot_vec[i] = 1
            vocab[f][v] = onehot_vec
    return vocab

def get_ID_vocab(meta_df, features):
    print('build vocab for encoded ID features')
    vocab = {}
    for f in tqdm(features):
        value_list = list(meta_df[f].unique())
        vocab[f] = {v:i+1 for i,v in enumerate(value_list)}
    return vocab

def get_multiID_vocab(meta_df, features, sep = ','):
    print('build vocab for encoded ID features')
    vocab = {}
    for f in features:
        print(f'\t{f}:')
        ID_freq = {}
        for row in tqdm(meta_df[f]):
            IDs = str(row).split(sep)
            for ID in IDs:
                if ID not in ID_freq:
                    ID_freq[ID] = 1
                else:
                    ID_freq[ID] += 1
        v_list = list(ID_freq.keys())
        vocab[f] = {v:i+1 for i,v in enumerate(v_list)}
    return vocab


def show_batch(batch):
    for k, batch in batch.items():
        if torch.is_tensor(batch):
            print(f"{k}: size {batch.shape}, \n\tfirst 5 {batch[:5]}")
        else:
            print(f"{k}: {batch}")
            

def wrap_batch(batch, device):
    '''
    Build feed_dict from batch data and move data to device
    '''
    for k,val in batch.items():
        if type(val).__module__ == np.__name__:
            batch[k] = torch.from_numpy(val)
        elif torch.is_tensor(val):
            batch[k] = val
        elif type(val) is list:
            batch[k] = torch.tensor(val)
        else:
            continue
        if batch[k].type() == "torch.DoubleTensor":
            batch[k] = batch[k].float()
        batch[k] = batch[k].to(device)
    return batch


############################################
#              Model Related               #
############################################

def init_weights(m):
    if 'Linear' in str(type(m)):
#         nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.xavier_normal_(m.weight, gain=1.)
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0.0, std=0.01)
    elif 'Embedding' in str(type(m)):
#         nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.xavier_normal_(m.weight, gain=1.0)
        print("embedding: " + str(m.weight.data))
        with torch.no_grad():
            m.weight[m.padding_idx].fill_(0.)
    elif 'ModuleDict' in str(type(m)):
        for param in module.values():
            nn.init.xavier_normal_(param.weight, gain=1.)
            with torch.no_grad():
                param.weight[param.padding_idx].fill_(0.)
                
                
def get_regularization(*modules):
    reg = 0
    for m in modules:
        for p in m.parameters():
            reg = torch.mean(p * p) + reg
    return reg
                
                
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
            
def sample_categorical_action(action_prob, candidate_ids, slate_size, with_replacement = True, 
                              batch_wise = False, return_idx = False):
    '''
    @input:
    - action_prob: (B, L)
    - candidate_ids: (B, L) or (1, L)
    - slate_size: K
    - with_replacement: sample with replacement
    - batch_wise: do batch wise candidate selection 
    '''
    if with_replacement:
        indices = Categorical(action_prob).sample(sample_shape = (slate_size,))
        indices = torch.transpose(indices, 0, 1)
    else:
        indices = torch.cat([torch.multinomial(prob, slate_size, replacement = False).view(1,-1) \
                             for prob in action_prob], dim = 0)
    action = torch.gather(candidate_ids,1,indices) if batch_wise else candidate_ids[indices]
    if return_idx:
        return action.detach(), indices.detach()
    else:
        return action.detach()


            
#######################################
#              Learning               #
#######################################

class LinearScheduler(object):
    '''
    Code used in DQN: https://github.com/dxyang/DQN_pytorch/blob/master/utils/schedules.py
    '''
    
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
    
    
class SinScheduler(object):
    '''
    Code used in DQN: https://github.com/dxyang/DQN_pytorch/blob/master/utils/schedules.py
    '''
    
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = np.sin(min(float(t) / self.schedule_timesteps, 1.0) * np.pi * 0.5)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
    
    