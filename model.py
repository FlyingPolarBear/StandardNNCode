'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-25 23:55:26
LastEditors: Derry
LastEditTime: 2021-08-10 19:19:30
Description: Standard model file of a neural network
'''

import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        n_hid = 64
        self.hidden = nn.Linear(args.n_in, n_hid)
        self.out = nn.Linear(n_hid, args.n_out)

    def forward(self, X, y):
        X = self.hidden(X)
        X = nn.functional.relu(X)
        y_out = self.out(X)

        poss = nn.functional.softmax(y_out, dim=1)
        y_pred = torch.max(poss, 1)[1]
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(y_out, y)
        return y_pred, loss

    def evaluate(self, X, y):
        self.eval()
        y_pred, loss = self.forward(X, y)
        acc = ((y_pred == y).int().sum().float() /
               float(y.shape[0])).cpu().numpy()
        return loss, acc


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
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2*2*64, 100)
        self.mlp2 = torch.nn.Linear(100, 10)

    def forward(self, x, y):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        y_out = self.mlp2(x)

        poss = nn.functional.softmax(y_out, dim=1)
        y_pred = torch.max(poss, 1)[1]
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(y_out, y)
        return y_pred, loss

    def evaluate(self, X, y):
        self.eval()
        y_pred, loss = self.forward(X, y)
        acc = ((y_pred == y).int().sum().float() /
               float(y.shape[0])).cpu().numpy()
        return loss, acc