# coding:utf-8

import numpy as np

def fc_forward(X, W, b):
    out = X @ W + b # np.matmul(x, w)
    cache = (W, X)
    return out, cache

def fc_backward(dout, cache):
    w, h = cache

    dW = h.T @ dout
    db = np.sum(dout, axis=0)
    dX = dout @ w.T

    return dX, dW, db

def relu_forward(x):
    out = np.maximum(x, 0) #compare with 0
    cache = x
    return out, x

def relu_backward(dout, cache):
    dX = dout.copy()
    dX[cache<=0] = 0
    return dX

# leaky relu
def lrelu_forward(x, a=1e-3):
    out = np.maximum(a*x, x)
    cache = (x, a)
    return out, cache

def lrelu_backward(dout, cache):
    x, a = cache
    dx = dout.copy()
    dx[x<0] *= a
    return dx



if __name__ == '__main__':
    pass


