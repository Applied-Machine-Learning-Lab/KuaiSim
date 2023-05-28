#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2019/12/2 下午7:38
import click

from GAN_SD.Gan_SD import GanSDModel
from data_loader import load_dataset
from utils.utils import *


@click.command()
@click.option('--dataset_path', type=click.Path('r'), default='./data/dataset.txt')
@click.option('--dim_seed', type=int, default=128, help='Seed dimension for Generator of GAN-SD')
@click.option('--batch_size', type=int, default=128, help='Batch size for GAN-SD')
@click.option('--learning_rate_generator', 'lr_g', type=float, default=0.001, help='Learning rate for Generator')
@click.option('--learning_rate_discriminator', 'lr_d', type=float, default=0.0001,
              help='Learning rate for Discriminator')
@click.option('--alpha', type=float, default=1.0, help='Coefficient for Entropy Loss')
@click.option('--beta', type=float, default=1.0, help='Coefficient for KL divergence')
@click.option('--seed', type=int, default=2019, help='Random seed for reproduce')
def main(dataset_path, dim_seed, batch_size, lr_g, lr_d, alpha, beta, seed):
    """
    Train GAN-SD for generating human-like user features
    """
    dim_user = 118
    dim_seed = dim_seed
    # expert_user_features = load_dataset(dataset_path)[0]
    expert_user_features = torch.load('user_feature')

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = GanSDModel(dim_user, dim_seed, lr_g, lr_d, expert_user_features, batch_size=batch_size, alpha=alpha,
                       beta=beta)
    model.train()
    model.save_model()


if __name__ == '__main__':
    main()
