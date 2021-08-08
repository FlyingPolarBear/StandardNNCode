'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-27 17:05:23
LastEditors: Derry
LastEditTime: 2021-08-08 17:21:46
Description: Standard utils file of a neural network
'''

def load_data(args):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed)
    return X_train, X_test, y_train, y_test
