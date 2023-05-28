#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2019/12/2 下午9:09
from utils.utils import *

dataset_path = r'data/dataset.txt'


def load_dataset(path=None):
    features, labels, clicks = [], [], []
    if path is None:
        path = dataset_path
    with open(path, 'r') as file:
        for line in file:
            features_l, labels_l, clicks_l = line.split('\t')
            features.append([float(x) for x in features_l.split(',')])
            labels.append([float(x) for x in labels_l.split(',')])
            clicks.append(int(clicks_l))
    features, labels, clicks = FLOAT(features), FLOAT(labels), FLOAT(clicks)

    return features, labels, clicks


def load_expert_trajectories(path=None):
    if path is None:
        path = dataset_path

    expert_trajectories = []
    with open(path, 'r') as file:
        cur_trajectory = []

        for line in file:
            line = line.replace('\t', ',')
            data = line.split(',')
            cur_page_index = data[90]
            if float(cur_page_index) == 1.0 and len(cur_trajectory) > 0:
                expert_trajectories.append(FLOAT(cur_trajectory))
                cur_trajectory = []

            data = data[0:88] + data[91:91 + 27] + [data[90]] + [data[88]]
            cur_trajectory.append(([float(x) for x in data]))
            

    return expert_trajectories


if __name__ == '__main__':
    # load_dataset()
    expert_data = load_expert_trajectories()