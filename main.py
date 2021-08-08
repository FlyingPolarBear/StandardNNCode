'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-25 23:39:03
LastEditors: Derry
LastEditTime: 2021-08-08 16:57:38
Description: Standard main file of a neural network
'''

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import Net
from utils import load_data


def train(my_model, train_loader, test_loader, args):
    for epoch in range(args.epoch):
        for batch, (X_train, y_train) in enumerate(train_loader):
            start = time.time()
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
            print("Epoch {:4d}".format(epoch),
                  "time= {:.2f}s".format(time.time()-start), end=' ')
            test(my_model, test_loader, args)


def test(my_model, test_loader, args):
    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(test_loader):
            if args.cuda:
                X_test = X_test.cuda()
                y_test = y_test.cuda()
            loss, acc = my_model.evaluate(X_test, y_test)

        print("Test set results:",
              "loss= {:.4f}".format(loss),
              "accuracy= {:.4f}".format(acc))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true',
                        default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_in', type=int, default=100)
    parser.add_argument('--n_out', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    X_train, X_test, y_train, y_test = load_data(args)
    args.n_in = X_train.shape[1]
    args.n_out = len(set(list(y_train)))

    # Construct data loader
    train_dataset = TensorDataset(torch.tensor(
        X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(
        X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True)

    # Model, optimizer and scheduler
    my_model = Net(args)
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

    train(my_model, train_loader, test_loader, args)
    test(my_model, test_loader, args)
