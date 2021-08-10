'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-27 17:05:23
LastEditors: Derry
LastEditTime: 2021-08-10 19:25:46
Description: Standard utils file of a neural network
'''
import torch

def load_mnist_data(args):
    from torchvision import datasets, transforms
    train_dataset = datasets.MNIST('./data', train=True,
                                   transform=transforms.ToTensor(),
                                   download=False)
    test_dataset = datasets.MNIST('./data', train=False,
                                  transform=transforms.ToTensor(),
                                  download=False)
    X_train = torch.tensor(train_dataset.data, dtype=torch.float32)
    y_train = torch.tensor(train_dataset.targets, dtype=torch.long)
    X_test = torch.tensor(test_dataset.data, dtype=torch.float32)
    y_test = torch.tensor(test_dataset.targets, dtype=torch.long)
    return X_train, y_train, X_test, y_test


def load_iris_data(args):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    load_mnist_data('')
