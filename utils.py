'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-27 17:05:23
LastEditors: Derry
LastEditTime: 2021-08-25 23:54:57
Description: Standard utils file of a neural network
'''
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def load_mnist_data(args):
    from torchvision import datasets, transforms
    train_dataset = datasets.FashionMNIST(args.data_path, train=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,))]),
                                          download=False)
    test_dataset = datasets.FashionMNIST(args.data_path, train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))]),
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
        X, y, test_size=0.1, random_state=args.seed)
    return (torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long),
            torch.as_tensor(X_test, dtype=torch.float32),
            torch.as_tensor(y_test, dtype=torch.long))


def plot(loss, acc, args):
    sns.set()
    plt.figure('loss')
    plt.title("loss of each epoch")
    plt.xlim(0, args.epoch)
    plt.plot(loss, color='b', label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(args.tmp_path+'/loss.png')

    plt.figure('acc')
    plt.title("accurancy of each epoch")
    plt.xlim(0, args.epoch)
    plt.ylim(min(acc)//10*10, 100)
    plt.plot(acc, color='b', label='accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(args.tmp_path+'/acc.png')


def load_pretrained(my_model, optimizer, scheduler, args):
    state = torch.load(args.model_path)
    my_model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    epoch = state['epoch']
    best_acc = state['accuracy']
    print("loaded pretrained model: epoch{:3d} acc= {:.2f}".format(
        epoch, best_acc))
    return my_model, optimizer, scheduler, epoch+1, best_acc


if __name__ == "__main__":
    load_mnist_data('')
