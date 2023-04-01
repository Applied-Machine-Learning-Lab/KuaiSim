import numpy as np
import pandas as pd
from tqdm import tqdm

from reader.KRMBSeqReader import KRMBSeqReader
from utils import padding_and_clip, get_onehot_vocab, get_multihot_vocab

class KRMBSeqReader_SubFB(KRMBSeqReader):
    '''
    KuaiRand Multi-Behavior Data Reader with subset of feedback types
    '''
    
    @staticmethod
    def parse_data_args(parser):
        '''
        args:
        - from KRMBSeqReader:
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
        parser = KRMBSeqReader.parse_data_args(parser)
        parser.add_argument('--feedback_subset', type=int, nargs='+', default=[0,1,2], 
                            help='indices of selected feedback types')
        return parser
        
        
    def __init__(self, args):
        '''
        - from KRMBSeqReader:
            - max_hist_seq_len
            - val_holdout_per_user
            - test_holdout_per_user
            - from BaseReader:
                - phase
                - n_worker
        '''
        super().__init__(args)
        self.feedback_subset = args.feedback_subset
        
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
        super()._read_data(args)
        
        # response meta
        self.response_list = [self.response_list[i] for i in args.feedback_subset]
        self.response_dim = len(self.response_list)
        self.padding_response = {resp: 0. for i,resp in enumerate(self.response_list)}
        self.response_neg_sample_rate = {k:self.response_neg_sample_rate[k] for k in self.response_list}
            