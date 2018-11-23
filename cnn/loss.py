# coding:utf-8

import numpy as np


def onehot(labels):
    y = np.zeros((labels.size, np.max(labels)+1))
    y[range(labels.size), labels] = 1
    return y


def softmax(x):
    # be careful about T ==> (3,2) and (2,)
    e_x = np.exp((x.T - np.max(x, axis=1)).T)
    return (e_x.T / e_x.sum(axis=1)).T


def regularization(model, reg_type='l2', lam=1e-3):
    pass


def cross_entropy(model, y_pred, y_train, lam=1e-3):
    m = y_pred.shape[0]

    # y_train is one-hot
    # y_train = onehot(y_train)
    prob = softmax(y_pred)
    log_like = -np.log(prob[range(m), y_train] + 1e-20)

    data_loss = np.sum(log_like) / m
    #reg_loss = regularization(model, reg_type='l2', lam=lam)

    return data_loss #+ reg_loss

# https://blog.csdn.net/Gipsy_Danger/article/details/81292148
def dcross_entropy(y_pred, y_train):
    m = y_pred.shape[0]

    grad_y = softmax(y_pred)
    grad_y[range(m), y_train] -= 1
    grad_y /= m
    return grad_y

if __name__ == '__main__':
    x = np.array([[1,5,10],[9,6,3]])
    print(softmax(x))