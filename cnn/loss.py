# coding:utf-8

import numpy as np

def softmax(x):
    # be careful about T ==> (3,2) and (2,)
    e_x = np.exp((x.T - np.max(x, axis=1)).T)
    return (e_x.T / e_x.sum(axis=1)).T

def regularization(model, reg_type='l2', lam=1e-3):
    pass



def cross_entorpy(model, y_pred, y_train, lam=1e-3):
    m = y_pred.shape[0]

    # y_train is one-hot
    prob = softmax(y_pred)
    log_like = -np.log(prob[range(m), y_train])

    data_loss = np.sum(log_like) / m
    #reg_loss = regularization(model, reg_type='l2', lam=lam)

    return data_loss #+ reg_loss

def dcross_entropy(y_pred, y_train):
    m = y_pred.shape[0]

    grad_y = softmax(y_pred)
    grad_y[range(m), y_train] -= 1
    grad_y /= m
    return grad_y

if __name__ == '__main__':
    x = np.array([[1,5,10],[9,6,3]])
    print(softmax(x))