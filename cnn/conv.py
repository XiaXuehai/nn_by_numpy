# coding:utf-8
'''
refer to : https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
'''

import numpy as np


def conv_forward(X, W, b, stride=1, padding=1):

    n_filter, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension')

    h_out, w_out = int(h_out), int(w_out)

    # 9x500
    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    # 20x9
    W_col = W.reshape(n_filter, -1)

    out = np.dot(W_col, X_col) + b
    out = out.reshape(n_filter, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, b, stride, padding, X_col)

    return out, cache

# same as caffe im2col
def im2col_indices(x, field_h, field_w, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_h, field_w, padding, stride)

    cols = x_padded[:, k, i, j]
    #print(cols.shape)
    C = x.shape[1]
    cols = cols.transpose(1,2,0).reshape(field_h*field_w*C, -1)
    #print(cols)
    return cols


def get_im2col_indices(x_shape, field_h, field_w, padding=1, stride=1):
    N, C, H, W = x_shape
    out_h = int((H + 2 * padding - field_h) / stride + 1)
    out_w = int((H + 2 * padding - field_w) / stride + 1)

    i0 = np.repeat(np.arange(field_h), field_w)
    i0 = np.tile(i0, C)
    i1 = stride * np.tile(np.arange(out_h), out_w)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    #print(i0)
    #print(i1)
    #print('i=',i)

    j0 = np.tile(np.arange(field_w), field_h*C)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    #print(j0)
    #print(j1)
    #print('j',j)

    k = np.repeat(np.arange(C), field_h*field_w).reshape(-1, 1)
    #print('k',k)

    return (k.astype(int), i.astype(int), j.astype(int))


def conv_backward(dout, cache):
    X, W, b, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape

    # it's weird
    # dout.shape = 5x20x10x10
    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)

    # in the network, the dout will be changed in back propagation
    dout_reshaped = dout.transpose(1,2,3,0).reshape(n_filter, -1) # 20x500
    dW = np.dot(dout_reshaped, X_col.T)
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1) # 20x9
    dX_col = np.dot(W_reshape.T, dout_reshaped) # 9x500
    # 9x500==>5x1x10x10
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

    return dX, dW, db

def col2im_indices(dX_col, x_shape, field_h=3, field_w=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=dX_col.dtype)
    k, i, j = get_im2col_indices(x_shape, field_h, field_w, padding, stride)
    cols_reshaped = dX_col.reshape(C * field_h * field_w, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


if __name__ == '__main__':
    # batch size : 5
    # input channel : 1
    # output :5*20*10*10
    x = np.random.rand(5,1,10,10)
    w = np.random.randn(20,1,3,3)
    b = np.random.rand(20,1)

    # forward
    out, cache = conv_forward(x, w, b)
    # backward
    dX, dW, db = conv_backward(out, cache)


