# coding:utf-8
'''
refer to : https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
           https://blog.csdn.net/Jason_yyz/article/details/80003271
'''

import numpy as np
from conv import *

def maxpool_forward(X, size=2, stride=2):
    def maxpool(X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    return _pool_forward(X, maxpool, size, stride)


def maxpool_backward(dout, cache):
    def dmaxpool(dX_col, dout_col, pool_cache):
        dX_col[pool_cache, range(dout_col.size)] = dout_col
        return dX_col

    return _pool_backward(dout, dmaxpool, cache)


def avgpool_forward(X, size=2, stride=2):
    def avgpool(X_col):
        out = np.mean(X_col, axis=0)
        cache = None
        return out, cache

    return _pool_forward(X, avgpool, size, stride)


def avgpool_backward(dout, cache):
    def davgpool(dX_col, dout_col, pool_cache):
        dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
        return dX_col

    return _pool_backward(dout, davgpool, cache)


def _pool_forward(X, pool_fun, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1 # no padding
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w) # 50×28×28
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride) # 4×9800

    out, pool_cache = pool_fun(X_col)

    # forward:reshape to the output size: 14x14x5x10
    out = out.reshape(h_out, w_out, n, d)
    out = out.transpose(2, 3, 0, 1) # 5x10x14x14

    cache = (X, size, stride, X_col, pool_cache)

    return out, cache


def _pool_backward(dout, dpool_fun, cache):
    X, size, stride, X_col, pool_cache = cache
    n, d, w, h = X.shape

    dX_col = np.zeros_like(X_col)
    # 5x10x14x14 => 14x14x5x10, then flattened to 1x9800
    # transpose step is necessary to get the correct arrangement
    dout_col = dout.transpose(2, 3, 0, 1).ravel()

    # maxpool: put the max value on the corresponding position of dX_col
    dX_col = dpool_fun(dX_col, dout_col, pool_cache)

    # We now have the stretched matrix of 4x9800, then undo it with col2im operation
    # dX would be 50x1x28x28
    dX = col2im_indices(dX_col, (n * d, 1, h, w), size, size, padding=0, stride=stride)
    # then reshape to
    dX = dX.reshape(X.shape)

    return dX


if __name__ == '__main__':
    # batch size : 5
    # input channel : 10
    x = np.random.rand(5,10,28,28)

    out, cache = maxpool_forward(x)

    dx = maxpool_backward(out, cache)






