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


def l2_reg(w, lam=1e-3):
    return 0.5 * lam * np.sum(w * w)


def dl2_reg(w, lam=1e-3):
    return lam * w


def regularization(model, reg_type='l2', lam=1e-3):
    reg_types = dict(
        l2=l2_reg
    )

    if reg_type not in reg_types:
        raise Exception('Regularization type is must "l1" or "l2".')

    reg_loss = 0
    for k in model.keys():
        if k[0]=='W':
            reg_loss += np.sum(reg_types[reg_type](model[k], lam))

    return reg_loss



def cross_entropy(model, y_pred, y_train, lam=1e-3):
    m = y_pred.shape[0]

    prob = softmax(y_pred)
    log_like = -np.log(prob[range(m), y_train] + 1e-20) # prevent to be 0

    data_loss = np.sum(log_like) / m
    reg_loss = regularization(model, reg_type='l2', lam=lam)

    return data_loss + reg_loss


# https://blog.csdn.net/Gipsy_Danger/article/details/81292148
def dcross_entropy(y_pred, y_train):
    m = y_pred.shape[0]

    grad_y = softmax(y_pred)
    grad_y[range(m), y_train] -= 1
    grad_y /= m  # why /m ?
    return grad_y

if __name__ == '__main__':
    x = np.array([[1,5,10],[9,6,3]])
    print(softmax(x))