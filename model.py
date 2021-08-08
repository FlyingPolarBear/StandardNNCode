'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-25 23:55:26
LastEditors: Derry
LastEditTime: 2021-08-08 16:57:17
Description: Standard model file of a neural network
'''

import torch
import torch.nn as nn


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
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
