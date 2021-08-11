'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-25 23:39:03
LastEditors: Derry
LastEditTime: 2021-08-11 21:24:59
Description: Standard main file of a neural network
'''

import os
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import *
from utils import *


def train(my_model, train_loader, test_loader, optimizer, scheduler, args, start_epoch=0):
    if args.pretrained and os.path.exists(args.model_path):
        my_model, optimizer, scheduler, start_epoch, best_acc = load_pretrained(
            my_model, optimizer, scheduler, args)

    test_loss_all, test_acc_all = [], []
    best_acc = 0

    for epoch in range(start_epoch, args.epoch):
        start = time.time()
        for batch, (X_train, y_train) in enumerate(train_loader):
            my_model.train()

            if args.cuda:
                X_train = X_train.cuda()
                y_train = y_train.cuda()

            optimizer.zero_grad()
            _, loss = my_model(X_train, y_train)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        if not args.fastmode:
            print("Epoch {:3d}".format(epoch),
                  "time= {:.2f} s".format(time.time()-start), end=' ')
            test_loss, test_acc = test(my_model, test_loader, args)

            if test_acc > best_acc:
                best_acc = test_acc
                state = {'model': my_model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'epoch': epoch,
                         'accuracy': test_acc}
                torch.save(state, args.model_path)

            test_loss_all.append(test_loss)
            test_acc_all.append(test_acc)
            plot(test_loss_all, test_acc_all, args)


def test(my_model, test_loader, args):
    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(test_loader):
            if args.cuda:
                X_test = X_test.cuda()
                y_test = y_test.cuda()
            loss, acc = my_model.evaluate(X_test, y_test)

        print("Test set results:",
              "loss= {:.4f}".format(loss),
              "accuracy= {:.2f} %".format(100*acc))
        return loss.item(), 100*acc.item()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true',
                        default=False, help='Validate during training pass.')
    parser.add_argument('--pretrained', action='store_true',
                        default=True, help='Using pretrained model parameter.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Number of samples in a batch.')
    parser.add_argument('--n_in', type=int, default=100)
    parser.add_argument('--n_out', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--data_path', type=str,
                        default="./data", help='Path of dataset')
    parser.add_argument('--tmp_path', type=str,
                        default="./tmp", help='Path of tmporary output')
    parser.add_argument('--model_path', type=str,
                        default="./model/best_model.tar",
                        help='Path of model parameter')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    X_train, y_train, X_test, y_test = load_mnist_data(args)
    args.n_in = X_train.shape[1]
    args.n_out = len(set(list(y_train)))

    # Construct data loader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True)

    # Model, optimizer and scheduler
    my_model = CNN(args)
    optimizer = torch.optim.AdamW(my_model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        print("cuda is available!")
        torch.cuda.manual_seed(args.seed)
        my_model.cuda()

    train(my_model, train_loader, test_loader, optimizer, scheduler, args)
