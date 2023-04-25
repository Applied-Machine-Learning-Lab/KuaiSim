import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
from tqdm import tqdm

def get_online_training_info(log_path, episode_features = [], training_losses = []):
    episode = []
    episode_report_list = {k: [] for k in episode_features}
    loss_report_list = {k: [] for k in training_losses}
    with open(log_path, 'r') as infile:
        for line in tqdm(infile):
            split = line.split('@')
            # episode
            episode.append(eval(split[0].split(':')[1]))
            # episode report
            episode_report = eval(split[1].strip()[len("online episode:"):])
            if len(episode_report_list) == 0:
                episode_report_list = {k:[v] for k,v in episode_report.items()}
            else:
                for k,L in episode_report_list.items():
                    L.append(episode_report[k])
            # loss report
            loss_report = eval(split[2].strip()[len("training:"):])
            if len(loss_report_list) == 0:
                loss_report_list = {k:[v] for k,v in loss_report.items()}
            else:
                for k,L in loss_report_list.items():
                    L.append(loss_report[k])
    info = {'episode': episode}
    info.update(episode_report_list)
    info.update(loss_report_list)
    return info

def smooth(values, window = 3):
    new_values = [np.mean(values[max(0,idx-window):min(idx+1,len(values))]) for idx in range(window, len(values))]
    return new_values

def multiplot_multiple_line(legend_names, list_of_stats, x_name, ncol = 2, row_height = 4):
    '''
    @input:
    - legend_names: [legend]
    - list_of_stats: [{field_name: [values]}]
    - x_name: x-axis field_name
    - ncol: number of subplots in each row
    '''
    plt.rcParams.update({'font.size': 14})
    assert ncol > 0
    features = list(list_of_stats[0].keys())
    features.remove(x_name)
    N = len(features)
    fig_height = 12 // ncol if len(features) == 1 else row_height*((N-1)//ncol+1)
    plt.figure(figsize = (16, fig_height))
    for i,field in enumerate(features):
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        minY,maxY = float('inf'),float('-inf')
        for j,L in enumerate(legend_names):
            X = list_of_stats[j][x_name]
            value_list = list_of_stats[j][field]
            minY,maxY = min(minY,min(value_list)),max(maxY,max(value_list))
            plt.plot(X[:len(value_list)], value_list, label = L)
        plt.ylabel(field)
        plt.xlabel(x_name)
        scale = 1e-4 + maxY - minY
        plt.ylim(minY - scale * 0.05, maxY + scale * 0.05)
        plt.legend()
    plt.show()

def plot_multiple_lines(list_of_stats, labels, 
                        fig_height = 4, font_size = 16, log_value = False):
    '''
    @input:
    - list_of_stats: [[x],[y]]
    - labels: [title_name]
    - ncol: number of subplots in each row
    - row_height: height of each row
    '''
    plt.rcParams.update({'font.size': font_size})
    N = len(list_of_stats)
    plt.figure(figsize = (16, fig_height))
    for i,stats in enumerate(list_of_stats):
        X,Y = stats
        plt.plot(X,Y,label = labels[i])
        if log_value:
            plt.yscale('log')
        plt.title(labels[i])
    plt.legend()
    plt.show()
    
def plot_multiple_bars(list_of_stats, features, 
                       ncol = 2, row_height = 4, font_size = 16, 
                       log_value = False, horizontal = False):
    '''
    @input:
    - list_of_stats: [[x],[x_name],[y]]
    - ncol: number of subplots in each row
    - row_height: height of each row
    '''
    plt.rcParams.update({'font.size': font_size})
    N = len(list_of_stats)
    assert ncol > 0 and len(features) == N
    fig_height = 12 // ncol if N == 1 else row_height*((N-1)//ncol+1)
    plt.figure(figsize = (16, fig_height))

    for i,stats in enumerate(list_of_stats):
        X,X_name,Y = stats
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        if horizontal:
            plt.barh(X,np.log(Y) if log_value else Y,label = features[i])
            plt.yticks(X,X_name)
            if log_value:
                plt.xlabel('freq in log')
        else:
            plt.bar(X,np.log(Y) if log_value else Y,label = features[i])
            plt.xticks(X,X_name)
            if log_value:
                plt.ylabel('freq in log')
        plt.title(features[i])
    plt.show()
    
def plot_multiple_hists(list_of_stats, features, 
                        ncol = 2, row_height = 4, font_size = 16, 
                        log_value = False, n_bin = 10):
    '''
    @input:
    - list_of_stats: [[y]]
    - ncol: number of subplots in each row
    - row_height: height of each row
    '''
    plt.rcParams.update({'font.size': font_size})
    N = len(list_of_stats)
    assert ncol > 0 and len(features) == N
    fig_height = 12 // ncol if N == 1 else row_height*((N-1)//ncol+1)
    plt.figure(figsize = (16, fig_height))
    for i,Y in enumerate(list_of_stats):
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        plt.hist(Y,label = features[i], bins = n_bin)
        plt.title(features[i])
        if log_value:
            plt.yscale('log')
    plt.show()