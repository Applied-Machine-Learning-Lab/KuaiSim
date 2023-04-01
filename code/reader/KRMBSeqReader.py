import numpy as np
import pandas as pd
from tqdm import tqdm

from reader.BaseReader import BaseReader
from utils import padding_and_clip, get_onehot_vocab, get_multihot_vocab

class KRMBSeqReader(BaseReader):
    '''
    KuaiRand Multi-Behavior Data Reader
    '''
    
    @staticmethod
    def parse_data_args(parser):
        '''
        args:
        - user_meta_file
        - item_meta_file
        - max_hist_seq_len
        - val_holdout_per_user
        - test_holdout_per_user
        - meta_file_sep
        - from BaseReader:
            - train_file
            - val_file
            - test_file
            - n_worker
        '''
        parser = BaseReader.parse_data_args(parser)
        parser.add_argument('--user_meta_file', type=str, required=True, 
                            help='user raw feature file_path')
        parser.add_argument('--item_meta_file', type=str, required=True, 
                            help='item raw feature file_path')
        parser.add_argument('--max_hist_seq_len', type=int, default=100, 
                            help='maximum history length in the sample')
        parser.add_argument('--val_holdout_per_user', type=int, default=5, 
                            help='number of holdout records for val set')
        parser.add_argument('--test_holdout_per_user', type=int, default=5, 
                            help='number of holdout records for test set')
        parser.add_argument('--meta_file_sep', type=str, default=',', 
                            help='separater of user/item meta csv file')
        return parser
        
    def log(self):
        super().log()
        print(f"\tval_holdout_per_user: {self.val_holdout_per_user}")
        print(f"\ttest_holdout_per_user: {self.test_holdout_per_user}")
        
    def __init__(self, args):
        '''
        - max_hist_seq_len
        - val_holdout_per_user
        - test_holdout_per_user
        - from BaseReader:
            - phase
            - n_worker
        '''
        print("initiate KuaiRandMultiBehaior sequence reader")
        self.max_hist_seq_len = args.max_hist_seq_len
        self.val_holdout_per_user = args.val_holdout_per_user
        self.test_holdout_per_user = args.test_holdout_per_user
        super().__init__(args)
        
    def _read_data(self, args):
        '''
        - log_data: pd.DataFrame
        - data: {'train': [row_id], 'val': [row_id], 'test': [row_id]}
        - users: [user_id]
        - user_id_vocab: {user_id: encoded_user_id}
        - user_meta: {user_id: {feature_name: feature_value}}
        - user_vocab: {feature_name: {feature_value: one-hot vector}}
        - selected_user_features
        - items: [item_id]
        - item_id_vocab: {item_id: encoded_item_id}
        - item_meta: {item_id: {feature_name: feature_value}}
        - item_vocab: {feature_name: {feature_value: one-hot vector}}
        - selected_item_features: [feature_name]
        - padding_item_meta: {feature_name: 0}
        - user_history: {uid: [row_id]}
        - response_list: [response_type]
        - padding_response: {response_type: 0}
        - 
        '''
        
        # read data_file
        print(f"Loading data files")
        self.log_data = pd.read_table(args.train_file, sep = args.data_separator)
        
        print("Load item meta data")
        item_meta_file = pd.read_csv(args.item_meta_file, sep = args.meta_file_sep)
        self.item_meta = item_meta_file.set_index('video_id').to_dict('index')
        print("Load user meta data")
        user_meta_file = pd.read_csv(args.user_meta_file, sep = args.meta_file_sep)
        self.user_meta = user_meta_file.set_index('user_id').to_dict('index')
        
        # user list, item list, user history
        self.users = list(self.log_data['user_id'].unique())
        self.items = list(self.log_data['video_id'].unique())
        self.user_history = {uid: list(self.log_data[self.log_data['user_id'] == uid].index) for uid in self.users}
        
        # id reindex
        self.user_id_vocab = {uid: i+1 for i,uid in enumerate(self.users)}
        self.item_id_vocab = {iid: i+1 for i,iid in enumerate(self.items)}
        
        # selected meta features
        self.selected_item_features = ['video_type', 'music_type', 'upload_type', 'tag']
        self.selected_user_features = ['user_active_degree', 'is_live_streamer', 'is_video_author', 
                                       'follow_user_num_range', 'fans_user_num_range', 
                                       'friend_user_num_range', 'register_days_range'] \
                                    + [f'onehot_feat{fid}' for fid in [0,1,6,9,10,11]]
        
        # meta feature vocabulary, {feature_name: {feature_value: one-hot/multi-hot vector}}
        self.user_vocab = get_onehot_vocab(user_meta_file, self.selected_user_features)
        self.item_vocab = get_onehot_vocab(item_meta_file, self.selected_item_features[:-1])
        self.item_vocab.update(get_multihot_vocab(item_meta_file, ['tag']))
        self.padding_item_meta = {f: np.zeros_like(list(v_dict.values())[0]) \
                                  for f, v_dict in self.item_vocab.items()}
        
        # response meta
        self.response_list = ['is_click', 'long_view', 'is_like', 'is_comment', 
                              'is_forward', 'is_follow', 'is_hate']
        self.response_dim = len(self.response_list)
        self.padding_response = {resp: 0. for i,resp in enumerate(self.response_list)}
        self.response_neg_sample_rate = self.get_response_weights()
        
        # {'train': [row_id], 'val': [row_id], 'test': [row_id]}
        self.data = self._sequence_holdout(args)
        
    def _sequence_holdout(self, args):
        '''
        Holdout validation and test set from log_data
        '''
        print(f"sequence holdout for users (-1, {args.val_holdout_per_user}, {args.test_holdout_per_user})")
        if args.val_holdout_per_user == 0 and args.test_holdout_per_user == 0:
            return {"train": self.log_data.index, "val": [], "test": []}
        data = {"train": [], "val": [], "test": []}
        for u in tqdm(self.users):
            sub_df = self.log_data[self.log_data['user_id'] == u]
            n_train = len(sub_df) - args.val_holdout_per_user - args.test_holdout_per_user
            if n_train < 0.6 * len(sub_df):
                continue
            data['train'].append(list(sub_df.index[:n_train]))
            data['val'].append(list(sub_df.index[n_train:n_train+args.val_holdout_per_user]))
            data['test'].append(list(sub_df.index[-args.test_holdout_per_user:]))
        for k,v in data.items():
            data[k] = np.concatenate(v)
        return data
    
    def get_response_weights(self):
        ratio = {}
        for f in self.response_list:
            counts = self.log_data[f].value_counts()
            ratio[f] = float(counts[1]) / counts[0]
        ratio['is_hate'] *= -1
        return ratio
            
    
    ###########################
    #        Iterator         #
    ###########################
        
    def __getitem__(self, idx):
        '''
        sample getter
        
        train batch after collate:
        {
            'user_id': (B,)
            'item_id': (B,)
            'is_click', 'long_view', ...: (B,)
            'uf_{feature}': (B,F_dim(feature)), user features
            'if_{feature}': (B,F_dim(feature)), item features
            'history': (B,max_H)
            'history_length': (B,)
            'history_if_{feature}': (B, max_H, F_dim(feature))
            'history_{response}': (B, max_H)
            'loss_weight': (B, n_response)
        }
        '''
        row_id = self.data[self.phase][idx]
        row = self.log_data.iloc[row_id]
        
        user_id = row['user_id'] # raw user ID
        item_id = row['video_id'] # raw item ID
        
        # user, item, responses
        record = {
            'user_id': self.user_id_vocab[row['user_id']], # encoded user ID
            'item_id': self.item_id_vocab[row['video_id']], # encoded item ID
        }
        for _,f in enumerate(self.response_list):
            record[f] = row[f]
        loss_weight = np.array([1. if record[f] == 1 else self.response_neg_sample_rate[f] \
                                for i,f in enumerate(self.response_list)])
        record["loss_weight"] = loss_weight
        
        # meta features
        user_meta = self.get_user_meta_data(user_id)
        record.update(user_meta)
        item_meta = self.get_item_meta_data(item_id)
        record.update(item_meta)
        
        # history features (max_H,)
        H_rowIDs = [rid for rid in self.user_history[user_id] if rid < row_id][-self.max_hist_seq_len:]
        history, hist_length, hist_meta, hist_response = self.get_user_history(H_rowIDs)
        record['history'] = np.array(history)
        record['history_length'] = hist_length
        for f,v in hist_meta.items():
            record[f'history_{f}'] = v
        for f,v in hist_response.items():
            record[f'history_{f}'] = v
            
        return record
    
    def get_user_meta_data(self, user_id):
        '''
        @input:
        - user_id: raw user ID
        @output:
        - user_meta_record: {'uf_{feature_name}: one-hot vector'}
        '''
        user_feature_dict = self.user_meta[user_id]
        user_meta_record = {f'uf_{f}': self.user_vocab[f][user_feature_dict[f]]\
                            for f in self.selected_user_features}
        return user_meta_record
    
    def get_item_meta_data(self, item_id):
        '''
        @input:
        - item_id: raw item ID
        @output:
        - item_meta_record: {'if_{feature_name}: one-hot vector'}
        '''
        item_feature_dict = self.item_meta[item_id]
        item_meta_record = {f'if_{f}': self.item_vocab[f][item_feature_dict[f]]\
                            for f in self.selected_item_features[:-1]}
        item_meta_record['if_tag'] = np.sum([self.item_vocab['tag'][tag_id] \
                                             for tag_id in item_feature_dict['tag'].split(',')], axis = 0)
        return item_meta_record
    
    def get_user_history(self, H_rowIDs):
        '''
        @input:
        - H_rowIDs: [idx of log_data]
        @output:
        - history: [encoded item ID]
        - L: history length (less than or equals to max_hist_seq_len)
        - hist_meta: {if_{feature_name}: (max_hist_seq_len, feature_dim)}
        - history_response: {response_type: (max_hist_seq_len,)}
        '''
        L = len(H_rowIDs)
        if L == 0:
            # (max_H,)
            history = [0] * self.max_hist_seq_len
            # {if_{feature_name}: (max_H, feature_dim)
            hist_meta = {f'if_{f}': np.tile(self.padding_item_meta[f],self.max_hist_seq_len) \
                         for f in self.selected_item_features}
            # {resp_type: (max_H)}
            history_response = {resp: np.array([self.padding_response[resp]] * self.max_hist_seq_len) \
                                for resp in self.response_list}
        else:
            H = self.log_data.iloc[H_rowIDs]
            # list of encoded iid
            item_ids = [self.item_id_vocab[iid] for iid in H['video_id']] 
            # (max_H,)
            history = padding_and_clip(item_ids, self.max_hist_seq_len) 
            # [{if_{feature}: one-hot vector}]
            meta_list = [self.get_item_meta_data(iid) for iid in H['video_id']] 
            # history item meta features: {if_{feature_name}: }
            hist_meta = {} 
            for f in self.selected_item_features:
                padding = [self.padding_item_meta[f] for i in range(self.max_hist_seq_len - L)]
                real_hist = [v_dict[f'if_{f}'] for v_dict in meta_list]
                # {if_{feature_name}: (max_H, feature_dim)}
                hist_meta[f'if_{f}'] = np.concatenate(padding + real_hist, axis = 0)
            # {resp_type: (max_H,)}
            history_response = {}
            for resp in self.response_list:
                padding = np.array([self.padding_response[resp]] * (self.max_hist_seq_len - L))
                real_resp = np.array(H[resp])
                history_response[resp] = np.concatenate([padding, real_resp], axis = 0)
        return history, L, hist_meta, history_response


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
        stats["n_item"] = len(self.items)
        stats["max_seq_len"] = self.max_hist_seq_len
        stats["user_features"] = self.selected_user_features
        stats["user_feature_dims"] = {f: len(list(v_dict.values())[0]) for f, v_dict in self.user_vocab.items()}
        stats["item_features"] = self.selected_item_features
        stats["item_feature_dims"] = {f: len(list(v_dict.values())[0]) for f, v_dict in self.item_vocab.items()}
        stats["feedback_type"] = self.response_list
        stats["feedback_size"] = self.response_dim
        stats["feedback_negative_sample_rate"] = self.response_neg_sample_rate
        return stats
