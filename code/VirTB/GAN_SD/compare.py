import matplotlib.pyplot as plt
import numpy as np
import torch

from GAN_SD.GeneratorModel import GeneratorModel
from data_loader import load_dataset

plt.style.use('seaborn-deep')

attribute_range = [8, 8, 11, 11, 11, 11, 2, 2, 3, 18, 3]
attribute_range_end = np.cumsum(attribute_range)
attribute_range_start = [x - y for x, y in zip(attribute_range_end, attribute_range)]


def histogram(attribute_index, real_distribution, virtual_distribution, sum):
    data_index = slice(attribute_range_start[attribute_index - 1], attribute_range_end[attribute_index - 1])
    real = real_distribution[data_index] / sum
    virtual = virtual_distribution[data_index] / sum

    bins = np.arange(attribute_range[attribute_index - 1]) + 1

    # plt.hist([real, virtual], bins, label=['real', 'virtual'])
    bar_width = 0.35
    plt.subplot(3, 4, attribute_index)
    plt.bar(bins - bar_width / 2, real, bar_width, align='center', label='real')
    plt.bar(bins + bar_width / 2, virtual, bar_width, align='center', label='virtual')

    plt.xticks(bins)
    plt.xlabel('Attribute value')
    plt.ylabel('Attribute %d' % (attribute_index, ))
    # plt.legend(loc='upper right')
    plt.legend()


if __name__ == '__main__':
    generator_model = GeneratorModel()
    generator_model.load_state_dict(torch.load('../model/user_G.pt'))

    real_users = load_dataset('../data/dataset.txt')[0][:, 0:88]

    m, _ = real_users.shape
    dim_seed = 128
    # 生成m个virtual user
    random_noise = torch.normal(torch.zeros(m, dim_seed),
                                torch.ones(m, dim_seed))
    virtual_users, _ = generator_model.generate(random_noise)

    real_distribution = real_users.cumsum(dim=0)[-1]
    virtual_distribution = virtual_users.cumsum(dim=0)[-1]

    # plot distribution over all features
    plt.figure(figsize=(20, 15))
    for i in range(1, len(attribute_range) + 1):
        histogram(i, real_distribution, virtual_distribution, m)
    plt.show()
