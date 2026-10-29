# -*- coding = utf-8 -*-
# @Time: 2025/2/11 22:01
# @Author: wisehone
# @File: data_loaders.py
# @SoftWare: PyCharm
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler
from utils.utils import weak_augment, strong_augment

# 注意：这里假设项目根目录下存在 data_contaminated_train 文件夹
CONTAM_ROOT = 'data_contaminated_train'

class SMDLoader(object):
    def __init__(self, data_path, win_size, step, mode='train', augment=None, w_jittering=0., scaling=0.,
                 s_jittering=0., time_wrapping=0., permutation=0., contamination_rate=0):
        # SMDLoader 暂时保持原样，如果SMD也有污染数据，逻辑同下
        self.win_size = win_size
        self.step = step
        self.mode = mode
        self.scale = StandardScaler()

        train_data = np.load(os.path.join(data_path, 'SMD_train.npy'))
        self.scale.fit(train_data)
        self.train_data = self.scale.transform(train_data)
        if augment is not None and augment == 'weak':
            self.train_data = weak_augment(self.train_data, w_jittering, scaling)
        elif augment is not None and augment == 'strong':
            self.train_data = strong_augment(self.train_data, self.win_size, s_jittering, time_wrapping, permutation)

        self.num_features = self.train_data.shape[1]

        test_data = np.load(os.path.join(data_path, 'SMD_test.npy'))
        self.test_data = self.scale.transform(test_data)
        if augment is not None and augment == 'weak':
            self.test_data = weak_augment(self.test_data, w_jittering, scaling)
        elif augment is not None and augment == 'strong':
            self.test_data = strong_augment(self.test_data, self.win_size, s_jittering, time_wrapping, permutation)

        self.test_label = np.load(os.path.join(data_path, 'SMD_test_label.npy'))

    def __len__(self):
        if self.mode == 'train':
            return (self.train_data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test_data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test_data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == 'train':
            return np.float32(self.train_data[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test_data[index:index + self.win_size]), np.float32(self.test_label[index:index + self.win_size])
        else:
            index = np.minimum(index, self.train_data.shape[0] - self.win_size - 1)
            return np.float32(self.train_data[index: index + self.win_size]), np.float32(np.zeros(self.win_size))

class MSLLoader(object):
    def __init__(self, data_path, win_size, step, mode='train', augment=None, w_jittering=0., scaling=0.,
                 s_jittering=0., time_wrapping=0., permutation=0., contamination_rate=0):
        self.win_size = win_size
        self.step = step
        self.mode = mode
        self.scale = StandardScaler()

        # --- 修改开始: 针对 MSL 处理污染数据逻辑 ---
        if contamination_rate > 0 and mode == 'train':
            # 构造污染数据的路径: data_contaminated_train/MSL/MSL_train_contam_r{rate}.npy
            # 注意：如果你的文件名对小数位有严格要求(如0.050)，请修改下面的格式化字符串，例如 f"r{contamination_rate:.3f}"
            file_name = f'MSL_train_contam_r{contamination_rate}.npy'
            train_path = os.path.join(CONTAM_ROOT, 'MSL', file_name)
            # 确保文件存在，否则报错或提示
            if not os.path.exists(train_path):
                 # 尝试一种补救，比如有的文件可能是 r0.050 而传入的是 0.05
                 alt_name = f'MSL_train_contam_r{contamination_rate:.3f}.npy' # 尝试3位小数
                 if os.path.exists(os.path.join(CONTAM_ROOT, 'MSL', alt_name)):
                     train_path = os.path.join(CONTAM_ROOT, 'MSL', alt_name)
            
            print(f"Loading Contaminated MSL Train Data: {train_path}")
            train_data = np.load(train_path)
        else:
            # 原始逻辑
            train_data = np.load(os.path.join(data_path, 'MSL_train.npy'))
        # --- 修改结束 ---

        self.scale.fit(train_data)
        self.train_data = self.scale.transform(train_data)
        if augment is not None and augment == 'weak':
            self.train_data = weak_augment(self.train_data, w_jittering, scaling)
        elif augment is not None and augment == 'strong':
            self.train_data = strong_augment(self.train_data, self.win_size, s_jittering, time_wrapping, permutation)

        self.num_features = self.train_data.shape[1]

        test_data = np.load(os.path.join(data_path, 'MSL_test.npy'))
        self.test_data = self.scale.transform(test_data)
        if augment is not None and augment == 'weak':
            self.test_data = weak_augment(self.test_data, w_jittering, scaling)
        elif augment is not None and augment == 'strong':
            self.test_data = strong_augment(self.test_data, self.win_size, s_jittering, time_wrapping, permutation)

        self.test_label = np.load(os.path.join(data_path, 'MSL_test_label.npy'))

    def __len__(self):
        if self.mode == 'train':
            return (self.train_data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test_data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test_data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == 'train':
            return np.float32(self.train_data[index: index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test_data[index: index + self.win_size]), np.float32(self.test_label[index: index + self.win_size])
        else:
            index = np.minimum(index, self.train_data.shape[0] - self.win_size - 1)
            return np.float32(self.train_data[index: index + self.win_size]), np.float32(np.zeros(self.win_size))

class SMAPLoader(object):
    def __init__(self, data_path, win_size, step, mode='train', augment=None, w_jittering=0., scaling=0.,
                 s_jittering=0., time_wrapping=0., permutation=0., contamination_rate=0):
        # SMAPLoader 保持原样，如需修改请参考 MSL
        self.win_size = win_size
        self.step = step
        self.mode = mode
        self.scale = StandardScaler()

        train_data = np.load(os.path.join(data_path, 'SMAP_train.npy'))
        self.scale.fit(train_data)
        self.train_data = self.scale.transform(train_data)
        if augment is not None and augment == 'weak':
            self.train_data = weak_augment(self.train_data, w_jittering, scaling)
        elif augment is not None and augment == 'strong':
            self.train_data = strong_augment(self.train_data, self.win_size, s_jittering, time_wrapping, permutation)

        self.num_features = self.train_data.shape[1]

        test_data = np.load(os.path.join(data_path, 'SMAP_test.npy'))
        self.test_data = self.scale.transform(test_data)
        if augment is not None and augment == 'weak':
            self.test_data = weak_augment(self.test_data, w_jittering, scaling)
        elif augment is not None and augment == 'strong':
            self.test_data = strong_augment(self.test_data, self.win_size, s_jittering, time_wrapping, permutation)

        self.test_label = np.load(os.path.join(data_path, 'SMAP_test_label.npy'))

    def __len__(self):
        if self.mode == 'train':
            return (self.train_data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test_data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test_data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == 'train':
            return np.float32(self.train_data[index: index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test_data[index: index + self.win_size]), np.float32(self.test_label[index: index + self.win_size])
        else:
            index = np.minimum(index, self.train_data.shape[0] - self.win_size - 1)
            return np.float32(self.train_data[index: index + self.win_size]), np.float32(np.zeros(self.win_size))


class SWaTLoader(object):
    def __init__(self, data_path, win_size, step, mode='train', augment=None, w_jittering=0., scaling=0.,
                 s_jittering=0., time_wrapping=0., permutation=0., contamination_rate=0):
        self.win_size = win_size
        self.step = step
        self.mode = mode
        self.scale = StandardScaler()

        # --- 修改开始: 针对 SWaT 处理污染数据逻辑 ---
        if contamination_rate > 0 and mode == 'train':
            # 原始文件名 swat_train2.csv -> swat_train2_contam_r{rate}.csv
            file_name = f'swat_train2_contam_r{contamination_rate}.csv'
            train_path = os.path.join(CONTAM_ROOT, 'SWAT', file_name) # 假设SWaT的文件夹名是SWAT
            
            # 文件名容错检查
            if not os.path.exists(train_path):
                 alt_name = f'swat_train2_contam_r{contamination_rate:.3f}.csv'
                 if os.path.exists(os.path.join(CONTAM_ROOT, 'SWAT', alt_name)):
                     train_path = os.path.join(CONTAM_ROOT, 'SWAT', alt_name)

            print(f"Loading Contaminated SWaT Train Data: {train_path}")
            train_data = pd.read_csv(train_path)
        else:
            train_data = pd.read_csv(os.path.join(data_path, 'swat_train2.csv'))
        # --- 修改结束 ---

        self.scale.fit(train_data.values[:, :-1])
        self.train_data = self.scale.transform(train_data.values[:, :-1])
        if augment is not None and augment == 'weak':
            self.train_data = weak_augment(self.train_data, w_jittering, scaling)
        elif augment is not None and augment == 'strong':
            self.train_data = strong_augment(self.train_data, self.win_size, s_jittering, time_wrapping, permutation)

        self.num_features = self.train_data.shape[1]

        test_data = pd.read_csv(os.path.join(data_path, 'swat2.csv'))
        self.test_data = self.scale.transform(test_data.values[:, :-1])
        if augment is not None and augment == 'weak':
            self.test_data = weak_augment(self.test_data, w_jittering, scaling)
        elif augment is not None and augment == 'strong':
            self.test_data = strong_augment(self.test_data, self.win_size, s_jittering, time_wrapping, permutation)

        self.test_label = test_data.values[:, -1:]

    def __len__(self):
        if self.mode == 'train':
            return (self.train_data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test_data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test_data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == 'train':
            return np.float32(self.train_data[index: index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test_data[index: index + self.win_size]), np.float32(self.test_label[index: index + self.win_size])
        else:
            index = np.minimum(index, self.train_data.shape[0] - self.win_size - 1)
            return np.float32(self.train_data[index: index + self.win_size]), np.float32(np.zeros(self.win_size))

class PSMLoader(object):
    def __init__(self, data_path, win_size, step, mode='train', augment=None, w_jittering=0., scaling=0.,
                 s_jittering=0., time_wrapping=0., permutation=0., contamination_rate=0):
        self.win_size = win_size
        self.step = step
        self.mode = mode
        self.scale = StandardScaler()

        # --- 修改开始: 针对 PSM 处理污染数据逻辑 ---
        if contamination_rate > 0 and mode == 'train':
            # 原始文件名 train.csv -> train_contam_r{rate}.csv
            file_name = f'train_contam_r{contamination_rate}.csv'
            train_path = os.path.join(CONTAM_ROOT, 'PSM', file_name)
            
            if not os.path.exists(train_path):
                 alt_name = f'train_contam_r{contamination_rate:.3f}.csv'
                 if os.path.exists(os.path.join(CONTAM_ROOT, 'PSM', alt_name)):
                     train_path = os.path.join(CONTAM_ROOT, 'PSM', alt_name)

            print(f"Loading Contaminated PSM Train Data: {train_path}")
            train_data = pd.read_csv(train_path)
        else:
            train_data = pd.read_csv(data_path + '/train.csv')
        # --- 修改结束 ---
        
        train_data = np.nan_to_num(train_data.values[:, 1:])
        self.scale.fit(train_data)
        self.train_data = self.scale.transform(train_data)
        if augment is not None and augment == 'weak':
            self.train_data = weak_augment(self.train_data, w_jittering, scaling)
        elif augment is not None and augment == 'strong':
            self.train_data = strong_augment(self.train_data, self.win_size, s_jittering, time_wrapping, permutation)

        self.num_features = self.train_data.shape[1]

        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = np.nan_to_num(test_data.values[:, 1:])
        self.test_data = self.scale.transform(test_data)
        if augment is not None and augment == 'weak':
            self.test_data = weak_augment(self.test_data, w_jittering, scaling)
        elif augment is not None and augment == 'strong':
            self.test_data = strong_augment(self.test_data, self.win_size, s_jittering, time_wrapping, permutation)


        self.test_label = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

    def __len__(self):
        if self.mode == 'train':
            return (self.train_data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test_data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test_data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == 'train':
            return np.float32(self.train_data[index: index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test_data[index: index + self.win_size]), np.float32(self.test_label[index: index + self.win_size])
        else:
            index = np.minimum(index, self.train_data.shape[0] - self.win_size - 1)
            return np.float32(self.train_data[index: index + self.win_size]), np.float32(np.zeros(self.win_size))

def get_data_loader(index, data_path, batch_size, win_size=1024, step=1, mode='train', dataset='SMD', num_workers=8,
                    augment=None, w_jittering=0., scaling=0.,
                    s_jittering=0., time_wrapping=0., permutation=0., contamination_rate=0):
    # 修改点：在函数定义中加入了 contamination_rate 参数，默认为 0
    
    if dataset == 'SMD':
        dataset = SMDLoader(data_path, win_size, step, mode, augment, w_jittering, scaling, s_jittering, time_wrapping, permutation, contamination_rate)
    elif dataset == 'MSL':
        dataset = MSLLoader(data_path, win_size, step, mode, augment, w_jittering, scaling, s_jittering, time_wrapping, permutation, contamination_rate)
    elif dataset == 'SMAP':
        dataset = SMAPLoader(data_path, win_size, step, mode, augment, w_jittering, scaling, s_jittering, time_wrapping, permutation, contamination_rate)
    elif dataset == 'PSM':
        dataset = PSMLoader(data_path, win_size, step, mode, augment, w_jittering, scaling, s_jittering, time_wrapping, permutation, contamination_rate)
    elif dataset == 'SWAT':
        dataset = SWaTLoader(data_path, win_size, step, mode, augment, w_jittering, scaling, s_jittering, time_wrapping, permutation, contamination_rate)

    shuffle = False
    if mode == 'train':
        shuffle = True

    sampler = None
    if mode == 'random':
        sampler = RandomSampler(data_source=dataset.train_data, replacement=True, num_samples=batch_size)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      drop_last=True)