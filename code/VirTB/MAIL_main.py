#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2019/12/2 下午7:38
import click

from MAIL.Mail import MailModel
from data_loader import load_expert_trajectories
from utils.utils import *


@click.command()
@click.option('--dataset_path', type=click.Path('r'), default='./data/dataset.txt')
# @click.option('--batch_size', type=int, default=128, help='Batch size for GAN-SD')
# @click.option('--learning_rate_generator', 'lr_g', type=float, default=0.001, help='Learning rate for Generator')
# @click.option('--learning_rate_discriminator', 'lr_d', type=float, default=0.0001,
#               help='Learning rate for Discriminator')
@click.option('--seed', type=int, default=2019, help='Random seed for reproduce')
def main(dataset_path, seed):
    """
    Train MAIL model
    """
    # expert_user_trajectory = load_expert_trajectories(dataset_path)
    # for i in range(len(expert_user_trajectory)):
    #     print(expert_user_trajectory[i][:,-1])
    # de
    expert_user_trajectory = torch.load('user_trajectory')
 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = MailModel(expert_user_trajectory)
    model.train()
    model.save_model()


if __name__ == '__main__':
    main()
