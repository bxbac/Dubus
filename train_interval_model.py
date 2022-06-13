#!/usr/bin/python
# coding:utf-8
"""
This module achieves to train interval estimation model
"""

import time
import os
import sys
import numpy as np
import pandas as pd

import paddle
import paddle.nn as nn

import paddle.fluid as fluid
import paddle.vision.transforms as T
#from paddle.static import InputSpec

class TestDataset(paddle.io.Dataset):
    """TestDataset
    """
    def __init__(self, x_test):
        """
        __init__
        :param x_test: input x_test
        :return:
        """
        self.x_test = x_test

    def __getitem__(self, idx):
        """
        __getitem__
        :param idx: idx
        :return: x_test
        """
        x_test = self.x_test[idx]

        return x_test

    def __len__(self):
        """
        __getitem__
        :param :
        :return: length of x_test
        """
        return len(self.x_test)


class TrainDataset(paddle.io.Dataset):
    """TrainDataset
    """
    def __init__(self, x_train, x_label):
        """
        __init__
        :param x_test: input x_test
        :param x_label: input x_label
        :return:
        """
        self.x_train = x_train
        self.x_label = x_label

    def __getitem__(self, idx):
        """
        __getitem__
        :param idx: idx
        :return: x_train, lable
        """
        x_train = self.x_train[idx]
        label = self.x_label[idx]

        return x_train, label

    def __len__(self):
        """
        __getitem__
        :param :
        :return: length of x_train
        """
        return len(self.x_train)


class RecommenderNet(nn.Layer):
    """RecommenderNet
    """
    def __init__(self, input_size, hidden_size):
        """
        __init__
        :param input_size: input size
        :param hidden_size: hidden size
        :return:
        """
        super(RecommenderNet, self).__init__()
        self.lstm = paddle.nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=3)
        self.lstm1 = paddle.nn.LSTM(input_size=64,
                                    hidden_size=hidden_size,
                                    num_layers=1,
                                    direction='bidirectional')
        self.lstm2 = paddle.nn.LSTM(input_size=128,
                                    hidden_size=64,
                                    num_layers=1)
        self.linear1 = paddle.nn.Linear(64, hidden_size)
        self.linear2 = paddle.nn.Linear(hidden_size, 32)
        self.linear3 = paddle.nn.Linear(32, hidden_size)
        self.outlinear = paddle.nn.Linear(hidden_size, 16)
        self.relu = paddle.nn.ReLU()


def forward(self, lstm_input):
        """
        forward
        :param lstm_input: lstm input
        :return:
        """
        x, (h, c) = self.lstm(lstm_input)
        x, (h, c) = self.lstm1(x)
        x, (h, c) = self.lstm2(x)
        h = paddle.reshape(h, [1, 1, -1])
        hidden = paddle.transpose(h, [1, 0, 2])
        x = self.linear1(hidden)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.relu(x)
        output = self.outlinear(x)
        output = paddle.squeeze(output)
        return output


class Trainer(object):
    """trainer
    """
    def __init__(self, train_data_path, test_data_path):
        """
        __init__
        :param train_data_path:  train data path
        :param test_data_path: test data path
        :return:
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def load_data(self):
        """
        Load_data
        :param:
        :return:
        """
        data_train = pd.read_csv(self.train_data_path, sep='\t', header=None)
        data_test = pd.read_csv(self.test_data_path, sep='\t', header=None)
        self.x_train = data_train.iloc[:, 2:-1]
        self.y_train = data_train.iloc[:, -1]

        self.x_train = self.x_train.iloc[0:100]
        self.y_train = self.y_train.iloc[0:100]
        self.x_test = data_test.iloc[:, 2:-1]
        self.y_test = data_test.iloc[:, -1]

        self.x_train = self.change_cols_format(self.x_train, 'feature')
        self.x_test = self.change_cols_format(self.x_test, 'feature')
        self.y_train = self.change_cols_format(self.y_train, 'label', 16)
        self.y_test = self.change_cols_format(self.y_test, 'label', 16)

        self.x_train = self.x_train.astype(np.float32)
        self.x_train = paddle.to_tensor(self.x_train)
        self.y_train = self.y_train.astype(np.float32)
        self.y_train = paddle.to_tensor(self.y_train)
        self.x_test = self.x_test.astype(np.float32)
        self.x_test = paddle.to_tensor(self.x_test)
        self.y_test = self.y_test.astype(np.float32)
        self.y_test = paddle.to_tensor(self.y_test)

    def train(self, feature_dim, hidden_size=64):
        """
        train
        :param:
        :return:
        """
        model = RecommenderNet(feature_dim, hidden_size)
        model = paddle.Model(model)

        optimizer = paddle.optimizer.Adam(parameters=model.parameters(), \
                                          learning_rate=0.0003)
        loss = paddle.nn.MSELoss()
        metric = paddle.metric.Accuracy()

        model.prepare(optimizer, loss, metric)
        train_dataset = TrainDataset(self.x_train, self.y_train)

        #
        train_loader = paddle.io.DataLoader(train_dataset, \
                                            return_list=True, \
                                            shuffle=True, \
                                            drop_last=True)

        model.fit(train_loader, epochs=5, verbose=1)
        test_dataset = TestDataset(self.x_test)
        # predict
        y_pred = model.predict(test_dataset)
        y_pred = paddle.to_tensor(y_pred)
        y_pred = paddle.transpose(y_pred, [1, 2, 0])
        # calc mae and rmse

    def change_cols_format(self, dataframe, symbol, dim=1):
        """
        change dataframe into np.array
        :param dataframe:
        :param symbol:  choose the column is label-col or feature-col
        :param dim: features dim
        :return:
        """
        arr = dataframe.values

        if symbol == 'feature':
            dim = len(arr[0][0].split(','))
            res_arr = np.zeros(shape=(len(arr), len(arr[0]), dim))
            for i in range(len(arr)):
                for j in range(len(arr[0])):
                    data_list = arr[i][j].split(',')
                    data_list = np.array(data_list)
                    data_list = data_list.astype(np.float)
                    for k in range(len(data_list)):
                        res_arr[i][j][k] = data_list[k]
        if symbol == 'label':
            res_arr = np.zeros(shape=(len(arr), dim, 1))
            for i in range(len(arr)):
                data_list = arr[i].split(',')
                for j in range(len(data_list)):
                    res_arr[i][j][0] = float(data_list[j])
    
        return res_arr


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Error argv: python train_main.py 1)train_data_path 2)test_data_path 3)feature_dim")
        sys.exit(1)

    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    feature_dim = sys.argv[3]

    trainer = Trainer(train_data_path, test_data_path)
    trainer.load_data()
    trainer.train(feature_dim)
