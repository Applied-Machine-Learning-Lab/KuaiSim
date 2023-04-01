import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from tqdm import tqdm

    
def worker_init_func(worker_id):
    worker_info = data.get_worker_info()
    worker_info.dataset.worker_id = worker_id
    
#############################################################################
#                              Dataset Class                                #
#############################################################################

class BaseReader(Dataset):
    
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--train_file', type=str, required=True, 
                            help='train data file_path')
        parser.add_argument('--val_file', type=str, default='', 
                            help='val data file_path')
        parser.add_argument('--test_file', type=str, default='', 
                            help='test data file_path')
        parser.add_argument('--n_worker', type=int, default=4,
                            help='number of worker for dataset loader')
        parser.add_argument('--data_separator', type=str, default='\t',
                            help='separator of csv file')
        return parser
    
    def log(self):
        print("Reader params:")
        print(f"\tn_worker: {self.n_worker}")
        for k,v in self.get_statistics().items():
            print(f"\t{k}: {v}")
            
    def __init__(self, args):
        '''
        - phase: one of ["train", "val", "test"]
        - data: {phase: pd.DataFrame}
        - data_fields: {field_name: (field_type, field_var)}
        - data_vocab: {field_name: {value: index}}
        '''
        self.phase = "train"
        self.n_worker = args.n_worker
        self._read_data(args)
        
    def _read_data(self, args):
        self.data = dict()
        print(f"Loading data files", end = '\r')
        self.data['train'] = pd.read_table(args.train_file, sep = args.data_separator)
        self.data['val'] = pd.read_table(args.val_file, sep = args.data_separator) \
                                if len(args.val_file) > 0 else self.data['train']
        self.data['test'] = pd.read_table(args.test_file, sep = args.data_separator) \
                                if len(args.test_file) > 0 else self.data['val']

    def get_statistics(self):
        return {'length': len(self)}
    
    def set_phase(self, phase):
        assert phase in ["train", "val", "test"]
        self.phase = phase
        
    def get_train_dataset(self):
        self.set_phase("train")
        return self
    
    def get_eval_dataset(self, phase = 'val'):
        self.set_phase(phase)
        return self
    
    def __len__(self):
        return len(self.data[self.phase])
    
    def __getitem__(self, idx):
        pass