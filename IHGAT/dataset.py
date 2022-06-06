import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
import scipy.io as sio
import numpy as np
import math
from numpy.random import shuffle, randint
from sklearn.preprocessing import OneHotEncoder
import os

class MtvDataset():
    def __init__(self, args):
        assert args.data in ['cub', 'orl', 'pie', '3source', 'politics', 'football'],  \
                            'This data not support.'
        self.args = args
        data_dct = {
            'cub': 'cub_googlenet_doc2vec_c10.mat',
            'orl': 'ORL_mtv.mat',
            'pie': 'PIE_face_10.mat',
            '3source': 'Tao-3source_double.mat',
            'politics': 'politicsie_9view.mat',
            'football': 'football_9view.mat'
        }
        self.data_type = args.data
        self.data_path = os.path.join('data', data_dct[self.data_type])
        self.missing_rate = args.missing_rate
        self.split_rate = args.split_rate
        self.train_set, self.test_set, self.view_number, self.all_sample_num = self.read_data()

    def get_data(self, data_type='train'):
        assert data_type in ['train', 'test', 'all']
        if data_type == 'train':
            data = self.train_set[0]
            label = self.train_set[1]
        elif data_type == 'test':
            data = self.test_set[0]
            label = self.test_set[1]
        data_info = []
        for i in range(self.view_number):
            data_info.append(data[i].shape)
        self.args.logger.info("The " + data_type + " views number is : " + str(self.view_number))
        self.args.logger.info("Each view shape is : " + str(data_info))
        return data, label

    def get_missing_mask(self):
        """
        Follow the CPM_Nets setting:
        https://github.com/hanmenghan/CPM_Nets/blob/master/util/util.py
        """
        one_rate = 1 - self.missing_rate
        alldata_len = self.all_sample_num
        view_num = self.view_number
        if one_rate <= (1 / view_num):
            enc = OneHotEncoder(categories=([range(view_num)] * 1))
            view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
            return torch.Tensor(view_preserve)
        if one_rate == 1:
            view_preserve = randint(1, 2, size=(alldata_len, view_num))
            return torch.Tensor(view_preserve)
        error = 1
        while error >= 0.005:
            # enc = OneHotEncoder(n_values=view_num) #  Passing 'n_values' is deprecated in version 0.20
            enc = OneHotEncoder(categories=([range(view_num)] * 1))
            view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
            one_num = view_num * alldata_len * one_rate - alldata_len
            ratio = one_num / (view_num * alldata_len)
            matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
            a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
            one_num_iter = one_num / (1 - a / one_num)
            ratio = one_num_iter / (view_num * alldata_len)
            matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
            matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
            ratio = np.sum(matrix) / (view_num * alldata_len)
            error = abs(one_rate - ratio)
        return torch.Tensor(matrix)

    def read_data(self):
        """
        Follow the CPM_Nets setting:
        https://github.com/hanmenghan/CPM_Nets/blob/master/util/util.py
        """
        ratio = self.split_rate

        data = sio.loadmat(self.data_path)

        self.args.logger.info('Load data from ' + self.data_path)

        view_number = data['X'].shape[1]

        X = np.split(data['X'], view_number, axis=1)
        X_train = []
        X_test = []
        labels_train = []
        labels_test = []
        if min(data['gt']) == 0:
            labels = data['gt'] + 1
        else:
            labels = data['gt']
        classes = max(labels)[0]
        all_length = 0
        for c_num in range(1, classes + 1):
            c_length = np.sum(labels == c_num)
            index = np.arange(c_length)
            shuffle(index)
            labels_train.extend(labels[all_length + index][0:math.floor(c_length * ratio)])
            labels_test.extend(labels[all_length + index][math.floor(c_length * ratio):])
            X_train_temp = []
            X_test_temp = []
            for v_num in range(view_number):
                X_train_temp.append(X[v_num][0][0].transpose()[all_length + index][0:math.floor(c_length * ratio)])
                X_test_temp.append(X[v_num][0][0].transpose()[all_length + index][math.floor(c_length * ratio):])
            if c_num == 1:
                X_train = X_train_temp
                X_test = X_test_temp
            else:
                for v_num in range(view_number):
                    X_train[v_num] = np.r_[X_train[v_num], X_train_temp[v_num]]
                    X_test[v_num] = np.r_[X_test[v_num], X_test_temp[v_num]]
            all_length = all_length + c_length

        train_data = {}
        test_data = {}
        if (self.args.normalize):
            for v_num in range(view_number):
                X_train[v_num] = self.normalize(X_train[v_num])
                X_test[v_num] = self.normalize(X_test[v_num])
                train_data[v_num] = torch.tensor(X_train[v_num])
                test_data[v_num] = torch.tensor(X_test[v_num])
        labels_train = torch.tensor(labels_train)
        labels_test = torch.tensor(labels_test)
        all_sample_num = torch.cat([labels_train, labels_test]).shape[0]
        return (train_data, labels_train), (test_data, labels_test), view_number, all_sample_num



    @staticmethod
    def normalize(data):
        """
        normalize data
        """
        m = np.mean(data)
        mx = np.max(data)
        mn = np.min(data)
        return (data - m) / (mx - mn)
