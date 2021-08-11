'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-27 17:05:23
LastEditors: Derry
LastEditTime: 2021-08-11 17:27:58
Description: Standard utils file of a neural network
'''
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def load_mnist_data(args):
    from torchvision import datasets, transforms
    train_dataset = datasets.MNIST(args.data_path, train=True,
                                   transform=transforms.ToTensor(),
                                   download=False)
    test_dataset = datasets.MNIST(args.data_path, train=False,
                                  transform=transforms.ToTensor(),
                                  download=False)
    return(torch.as_tensor(train_dataset.data, dtype=torch.float32),
           torch.as_tensor(train_dataset.targets, dtype=torch.long),
           torch.as_tensor(test_dataset.data, dtype=torch.float32),
           torch.as_tensor(test_dataset.targets, dtype=torch.long))


def load_iris_data(args):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed)
    return (torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long),
            torch.as_tensor(X_test, dtype=torch.float32),
            torch.as_tensor(y_test, dtype=torch.long))


def plot(loss, acc, args):
    sns.set()
    plt.figure('loss')
    plt.xlim(0, args.epoch)
    plt.plot(loss, color='b', label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(args.tmp_path+'/loss.png')

    plt.figure('acc')
    plt.xlim(0, args.epoch)
    plt.ylim(90, 100)
    plt.plot(acc, color='b', label='accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(args.tmp_path+'/acc.png')


if __name__ == "__main__":
    load_mnist_data('')
