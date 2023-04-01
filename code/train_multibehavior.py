from tqdm import tqdm
from time import time
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
from model.simulator import *
from reader import *
import os
import utils
from sklearn.metrics import roc_auc_score

def do_eval(model, reader, args):
    reader.set_phase("val")
    eval_loader = DataLoader(reader, batch_size = args.val_batch_size,
                             shuffle = False, pin_memory = False, 
                             num_workers = reader.n_worker)
    val_report = {'loss': [], 'auc': {}}
    Y_dict = {f: [] for f in model.feedback_types}
    P_dict = {f: [] for f in model.feedback_types}
    pbar = tqdm(total = len(reader))
    with torch.no_grad():
        for i, batch_data in enumerate(eval_loader):
            wrapped_batch = utils.wrap_batch(batch_data, device = args.device)
            out_dict = model.do_forward_and_loss(wrapped_batch)
            loss = out_dict['loss']
            val_report['loss'].append(loss.item())
            for j,f in enumerate(model.feedback_types):
                Y_dict[f].append(wrapped_batch[f].view(-1).detach().cpu().numpy())
                P_dict[f].append(out_dict['preds'][:,:,j].view(-1).detach().cpu().numpy())
            pbar.update(args.batch_size)
    val_report['loss'] = (np.mean(val_report['loss']), np.min(val_report['loss']), np.max(val_report['loss']))
    for f in model.feedback_types:
        val_report['auc'][f] = roc_auc_score(np.concatenate(Y_dict[f]), 
                                             np.concatenate(P_dict[f]))
    pbar.close()
    return val_report


if __name__ == '__main__':
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--reader', type=str, required=True, help='Data reader class')
    init_parser.add_argument('--model', type=str, required=True, help='User response model class.')
    initial_args, _ = init_parser.parse_known_args()
    print(initial_args)
    modelClass = eval('{0}.{0}'.format(initial_args.model))
    readerClass = eval('{0}.{0}'.format(initial_args.reader))
    
    # control args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=128, help='validation batch size')
    parser.add_argument('--test_batch_size', type=int, default=128, help='test batch size')
    parser.add_argument('--save_with_val', action='store_true', help='save when validation check is true')
    parser.add_argument('--epoch', type=int, default=10, help='number of epoch')
    parser.add_argument('--cuda', type=int, default=-1, help='cuda device number; set to -1 (default) if using cpu')
    
    # customized args
    parser = modelClass.parse_model_args(parser)
    parser = readerClass.parse_data_args(parser)
    args, _ = parser.parse_known_args()
    print(args)
    
    utils.set_random_seed(args.seed)
    
    reader = readerClass(args)
    print('data statistics:\n', reader.get_statistics())
    
    # cuda
    if args.cuda >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        torch.cuda.set_device(args.cuda)
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    args.device = device
    
    # model and optimizer
    model = modelClass(args, reader.get_statistics(), device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.optimizer = optimizer
    
    try:
        
        best_auc = {f: 0 for f in model.feedback_types}
        print(f"validation before training:")
        val_report = do_eval(model, reader, args)
        print(f"Val result:")
        print(val_report)
        
        epo = 0
        stop_count = 0
        while epo < args.epoch:
            epo += 1
            print(f"epoch {epo} training")
            
            # train an epoch
            model.train()
            reader.set_phase("train")
            train_loader = DataLoader(reader, batch_size = args.batch_size, 
                                      shuffle = True, pin_memory = True,
                                      num_workers = reader.n_worker)
            t1 = time()
            pbar = tqdm(total = len(reader))
            step_loss = []
            step_behavior_loss = {fb: [] for fb in model.feedback_types}
            for i, batch_data in enumerate(train_loader):
                optimizer.zero_grad()
                wrapped_batch = utils.wrap_batch(batch_data, device = device)
                if epo == 0 and i == 0:
                    utils.show_batch(wrapped_batch)
                out_dict = model.do_forward_and_loss(wrapped_batch)
                loss = out_dict['loss']
                loss.backward()
                step_loss.append(loss.item())
                for fb, v in out_dict['behavior_loss'].items():
                    step_behavior_loss[fb].append(v)
                optimizer.step()
                pbar.update(args.batch_size)
                if i % 100 == 0:
                    print(f"Iteration {i}, loss: {np.mean(step_loss[-100:])}")
                    print({fb: np.mean(v[-100:]) for fb,v in step_behavior_loss.items()})
            pbar.close()
            print("Epoch {}; time {:.4f}".format(epo, time() - t1))

            # validation
            t2 = time()
            print(f"epoch {epo} validating")
            val_report = do_eval(model, reader, args)
            print(f"Val result:")
            print(val_report)
            improve = 0
            for f,v in val_report['auc'].items():
                if v > best_auc[f]:
                    improve += 1
                    best_auc[f] = v

            # save model when no less than 50% of the feedback types are improved
            if args.save_with_val:
                if improve >= 0.5 * len(model.feedback_types):
                    model.save_checkpoint()
                    stop_count = 0
                else:
                    stop_count += 1
                if stop_count >= 3:
                    break
            else:
                model.save_checkpoint()
            
    except KeyboardInterrupt:
        print("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            print(os.linesep + '-' * 20 + ' END: ' + utils.get_local_time() + ' ' + '-' * 20)
            exit(1)
    
    
    