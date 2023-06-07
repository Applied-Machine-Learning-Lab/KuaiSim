import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import Namespace
from datetime import datetime as dt

from reader import *
from model.simulator import *
from utils import wrap_batch, show_batch


if __name__ == '__main__':
    
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--behavior_model_log_file', type=str, required=True, 
                             help='multi-behavior user model training log path')
    init_parser.add_argument('--data_output_path', type=str, required=True, 
                             help='output path for generated data')
    args, _ = init_parser.parse_known_args()
    print(args)

    # get model and reader
    with open(args.behavior_model_log_file, 'r') as infile:
        meta_args = eval(infile.readline())
        training_args = eval(infile.readline())
    print(meta_args)
    print(training_args)
    
    # cuda
    if training_args.cuda >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(training_args.cuda)
        torch.cuda.set_device(training_args.cuda)
        device = f"cuda:{training_args.cuda}"
    else:
        device = "cpu"
    training_args.device = device

    readerClass = eval('{0}.{0}'.format(meta_args.reader))
    modelClass = eval('{0}.{0}'.format(meta_args.model))
    reader = readerClass(training_args)
    model = modelClass(training_args, reader.get_statistics(), device)
    model.load_from_checkpoint(with_optimizer = False)

    df = reader.log_data

    print('#user:', len(df['user_id'].unique()), ' #item:', len(df['video_id'].unique()), ' #record:', len(df))

    model = model.to(device)
    model.device = device

    # generate session encoding
    user_id = []
    session_id = []
    session_enc = []
    return_day = []
    users = list(df['user_id'].unique())
    with torch.no_grad():
        pbar = tqdm(total = len(users))
        for i,uid in enumerate(users):
            user_hist = df[df['user_id'] == uid]
            user_dates = user_hist['date'].unique()
            B = len(user_dates)

            # get data
            records = {'history': [], 'history_length': []}
            records.update({f'history_if_{f}': [] for f in reader.selected_item_features})
            records.update({f'history_{f}': [] for f in reader.response_list})
            row_ids = []
            for d in user_dates:
                day_session = user_hist[user_hist['date'] == d]
                row_ids = row_ids + list(day_session.index)
                row_ids = row_ids[-50:]
                history, hist_length, hist_meta, hist_response = reader.get_user_history(row_ids)
                # (max_H,)
                records['history'].append(np.array(history))
                # scalar
                records['history_length'].append(np.array([hist_length]))
                for f,v in hist_meta.items():
                    # (max_H, feature_dim)
                    records[f'history_{f}'].append(v)
                for f,v in hist_response.items():
                    # (max_H, )
                    records[f'history_{f}'].append(v)
            # collate batch
            for k,v in records.items():
                if k == 'history_length':
                    records[k] = torch.tensor(v)
                elif k[:11] == 'history_if_':
                    # (B * max_H, feature_dim)
                    combinedV = np.concatenate(v, axis = 0)
                    # (B, max_H, feature_dim)
                    records[k] = torch.tensor(combinedV).view(B,reader.max_hist_seq_len,-1)
                else:
                    # (B * max_H,)
                    combinedV = np.concatenate(v, axis = 0)
                    # (B, max_H)
                    records[k] = torch.tensor(combinedV).view(B,reader.max_hist_seq_len)
            # (B,)
            records['user_id'] = torch.tensor([reader.user_id_vocab[uid]] * B)
            user_meta = reader.get_user_meta_data(uid)
            for k,v in user_meta.items():
                # (B, feature_dim)
                records[k] = torch.tensor(v).view(1,-1).tile((B,1))

            # get session encoding
            wrapped_batch = wrap_batch(records, device = device)
            if i == 0:
                show_batch(wrapped_batch)
            out_dict = model.encode_state(wrapped_batch, B)
            pbar.update(1)

            user_id.append(records['user_id'][:-1].detach())
            session_id.append(torch.arange(1,B).detach())
            session_enc.append(out_dict['state'][:-1].detach())
            T = [dt.strptime(str(d), "%Y%m%d") for d in user_dates]
            return_day = return_day + [(T[i+1] - T[i]).days for i in range(B-1)]
        pbar.close()

    # data dict
    sess_df_dict = {
        'user_id': torch.cat(user_id, dim = 0).cpu().numpy(),
        'session_id': torch.cat(session_id, dim = 0).cpu().numpy(),
        'return_day': np.array(return_day)
    }
    session_enc = torch.cat(session_enc, dim = 0).cpu().numpy()
    print('encoding data shape: ', session_enc.shape)
    for dim in range(session_enc.shape[1]):
        sess_df_dict[f'session_enc_{dim}'] = [v for v in session_enc[:,dim]]
    sess_df = pd.DataFrame.from_dict(sess_df_dict)
    sess_df.to_csv(args.data_output_path, index = False)