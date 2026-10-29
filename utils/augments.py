# -*- coding = utf-8 -*-
# @Time: 2025/2/13 22:32
# @Author: wisehone
# @File: augments.py
# @SoftWare: PyCharm
import numpy as np
import tsaug

class jitter(object):
    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, x):
        return x + np.random.normal(loc=0., scale=self.sigma, size=x.shape)

class scale(object):
    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, x):
        factors = np.random.normal(loc=1., scale=self.sigma, size=x.shape)
        res = x * factors
        return res

class time_wrap(object):
    def __init__(self, n_speed_change=100, max_speed_ratio=10):
        # print('hh ', type(n_speed_change))
        self.transform = tsaug.TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)

    def __call__(self, x):
        return self.transform.augment(x)

class permutation_f(object):
    def __init__(self, win_size):
        self.win_size = win_size

    def __call__(self, x):
        # shuffle each time point in every window
        batch_size = x.shape[0] // self.win_size
        x_copy = x.copy()

        for i in range(batch_size):
            # print('before')
            # print(x[i * self.win_size:(i + 1) * self.win_size])
            np.random.shuffle(x_copy[i * self.win_size:(i + 1) * self.win_size])
            # print('after')
            # print(x[i * self.win_size:(i + 1) * self.win_size])

        if batch_size * self.win_size < x.shape[0]:
            np.random.shuffle(x_copy[batch_size * self.win_size:])

        return x_copy



