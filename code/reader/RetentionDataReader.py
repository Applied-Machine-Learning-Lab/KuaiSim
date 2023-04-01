import numpy as np
import pandas as pd
from tqdm import tqdm

from reader.BaseReader import BaseReader
from utils import padding_and_clip, get_onehot_vocab, get_multihot_vocab

class RetentionDataReader(BaseReader):
    '''
    Session Encoding Data Reader
    Data format: (user_id, session_id, session_enc, return_day)
    '''
    
    @staticmethod
    def parse_data_args(parser):
        '''
        args:
        --max_sess_seq_len
        --val_holdout_per_user
        --test_holdout_per_user
        - from BaseReader:
            - train_file
            - val_file
            - test_file
            - n_worker
        '''
        parser = BaseReader.parse_data_args(parser)
        parser.add_argument('--max_sess_seq_len', type=int, default=100, 
                            help='maximum history length in the input sequence')
        parser.add_argument('--max_return_day', type=int, default=10, 
                            help='number of possible return_day for classification')
        parser.add_argument('--val_holdout_per_user', type=int, default=1, 
                            help='number of holdout records for val set')
        parser.add_argument('--test_holdout_per_user', type=int, default=1, 
                            help='number of holdout records for test set')
        return parser
        
    def log(self):
        super().log()
        
    def __init__(self, args):
        '''
        - max_hist_seq_len
        - val_holdout_per_user
        - test_holdout_per_user
        - from BaseReader:
            - phase
            - n_worker
            
        '''
        print("initiate RetentionDataReader sequence reader")
        self.max_sess_seq_len = args.max_sess_seq_len
        self.max_return_day = args.max_return_day
        self.val_holdout_per_user = args.val_holdout_per_user
        self.test_holdout_per_user = args.test_holdout_per_user
        super().__init__(args)
        
    def _read_data(self, args):
        '''
        - log_data: pd.DataFrame
        - data: {'train': [row_id], 'val': [row_id], 'test': [row_id]}
        - users: [user_id]
        - user_id_vocab: {user_id: encoded_user_id}
        - user_history: {uid: [row_id]}
        '''
        
        # read data_file
        print(f"Loading data files")
        self.log_data = pd.read_csv(args.train_file)
        self.log_data[self.log_data['return_day'] > 10] = 10
        
        self.users = list(self.log_data['user_id'].unique())
        self.user_id_vocab = {uid: i+1 for i,uid in enumerate(self.users)}
        
        self.user_history = {uid: list(self.log_data[self.log_data['user_id'] == uid].index) for uid in self.users}
        
        self.enc_dim = len([col for col in self.log_data if 'session_enc_' in col])
        self.padding_return_day = 10
        
        # {'train': [row_id], 'val': [row_id], 'test': [row_id]}
        self.data = self._sequence_holdout(args)
        
    def _sequence_holdout(self, args):
        print(f"sequence holdout for users (-1, {args.val_holdout_per_user}, {args.test_holdout_per_user})")
        data = {"train": [], "val": [], "test": []}
        for u in tqdm(self.users):
            sub_df = self.log_data[self.log_data['user_id'] == u]
            n_train = len(sub_df) - args.val_holdout_per_user - args.test_holdout_per_user
            if n_train < 0.8 * len(sub_df):
                continue
            data['train'].append(list(sub_df.index[:n_train]))
            data['val'].append(list(sub_df.index[n_train:n_train+args.val_holdout_per_user]))
            data['test'].append(list(sub_df.index[-args.test_holdout_per_user:]))
        for k,v in data.items():
            data[k] = np.concatenate(v)
        return data
            
    
    ###########################
    #        Iterator         #
    ###########################
        
    def __getitem__(self, idx):
        '''
        train batch after collate:
        {
            'user_id': (B,)
            'return_day': (B,)
            'history_encoding': (B, max_H, enc_dim)
            'history_response': (B, max_H)
            'history_length': (B,)
        }
        '''
        row_id = self.data[self.phase][idx]
        row = self.log_data.iloc[row_id]
        
        user_id = row['user_id'] # raw user ID
        
        # (max_H,)
        H_rowIDs = [rid for rid in self.user_history[user_id] if rid < row_id][-self.max_sess_seq_len:]
        # (max_H, enc_dim), scalar, (max_H,)
        hist_enc, hist_length, hist_response = self.get_user_history(H_rowIDs)
        # (enc_dim,)
        current_sess_enc = np.array([row[f'session_enc_{dim}'] for dim in range(self.enc_dim)])
        
        record = {
            'user_id': self.user_id_vocab[row['user_id']], # encoded user ID
            'sess_encoding': current_sess_enc,
            'return_day': int(row['return_day'])-1,
            'history_encoding': hist_enc,
            'history_response': hist_response,
            'history_length': hist_length
        }
        return record
    
    def get_user_history(self, H_rowIDs):
        L = len(H_rowIDs)
        if L == 0:
            # (max_H, enc_dim)
            history_encoding = np.zeros((self.max_sess_seq_len, self.enc_dim))
            # {resp_type: (max_H)}
            history_response = np.array([self.padding_return_day] * self.max_sess_seq_len)
        else:
            H = self.log_data.iloc[H_rowIDs]
            pad_hist_encoding = np.zeros((self.max_sess_seq_len - L, self.enc_dim))
            real_hist_encoding = [np.array(H[f'session_enc_{dim}']).reshape((-1,1)) for dim in range(self.enc_dim)]
            real_hist_encoding = np.concatenate(real_hist_encoding, axis = 1)
            history_encoding = np.concatenate((pad_hist_encoding, real_hist_encoding), axis = 0)
            pad_hist_response = np.array([self.padding_return_day] * (self.max_sess_seq_len - L))
            real_hist_response = np.array(H['return_day'])-1
            history_response = np.concatenate((pad_hist_response, real_hist_response), axis = 0)
        return history_encoding, L, history_response.astype(int)


    def get_statistics(self):
        '''
        - n_user
        - n_item
        - s_parsity
        - from BaseReader:
            - length
            - fields
        '''
        stats = {}
        stats["raw_data_size"] = len(self.log_data)
        stats["data_size"] = [len(self.data['train']), len(self.data['val']), len(self.data['test'])]
        stats["n_user"] = len(self.users)
        stats["max_sess_len"] = self.max_sess_seq_len
        stats["enc_dim"] = self.enc_dim
        stats["max_return_day"] = self.max_return_day
        return stats
