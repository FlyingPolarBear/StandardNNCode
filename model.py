'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-25 23:55:26
LastEditors: Derry
LastEditTime: 2021-08-26 01:03:14
Description: Standard model file of a neural network
'''

import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(args.n_in, args.n_hid)
        self.out = nn.Linear(args.n_hid, args.n_out)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(args.n_hid)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X, y):
        X = self.relu(self.hidden(X))
        X = self.batchnorm(X)
        X = self.dropout(X)
        y_out = self.out(X)

        poss = nn.functional.softmax(y_out, dim=1)
        y_pred = torch.max(poss, 1)[1]
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(y_out, y)
        return y_pred, loss

    def evaluate(self, X, y):
        self.eval()
        y_pred, loss = self.forward(X, y)
        acc = (y_pred == y).int().sum() / y.shape[0]
        return loss.item(), 100*acc.item()


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
        )
        self.fn1 = torch.nn.Sequential(
            torch.nn.Linear(2*2*64, args.n_hid),
            torch.nn.BatchNorm1d(args.n_hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
        )
        self.fn2 = torch.nn.Sequential(
            torch.nn.Linear(args.n_hid, args.n_out),
            torch.nn.BatchNorm1d(args.n_out),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
        )

    def forward(self, x, y):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fn1(x.view(x.size(0), -1))
        y_out = self.fn2(x)

        poss = nn.functional.softmax(y_out, dim=1)
        y_pred = torch.max(poss, 1)[1]
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(y_out, y)
        return y_pred, loss

    def evaluate(self, X, y):
        self.eval()
        y_pred, loss = self.forward(X, y)
        acc = (y_pred == y).int().sum() / y.shape[0]
        return loss.item(), 100*acc.item()
