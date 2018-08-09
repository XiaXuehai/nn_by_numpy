#https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
#######################################################################
import numpy as np


def conv_forward(X, W, b, stride=1, padding=1):

    n_filter, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filter, -1)

    out = np.dot(W_col, X_col) + b
    out = out.reshape(n_filter, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, b, stride, padding, X_col)

    return out, cache

def im2col_indices(x, field_h, field_w, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_h, field_w, padding, stride)

    cols = x_padded[:, k, i, j]
    print(cols.shape)
    C = x.shape[1]
    cols = cols.transpose(1,2,0).reshape(field_h*field_w*C, -1)

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
    print('i=',i)

    j0 = np.tile(np.arange(field_w), field_h*C)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    #print(j0)
    #print(j1)
    print('j',j)

    k = np.repeat(np.arange(C), field_h*field_w).reshape(-1, 1)
    print(k.shape)

    return (k.astype(int), i.astype(int), j.astype(int))



im2col_indices(np.random.rand(1,1,2,2), 3, 3)

##########################################################################
##########################################################################

def conv_backward(dout, cache):
    X, W, b, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape

    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)

    dout_reshaped = dout.transpose(1,2,3,0).reshape(n_filter, -1)
    dW = np.dot(dout_reshaped, X_col.T)
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = np.dot(W_reshape.T, dout_reshaped)
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

    return dX, dW, db

def col2im_indices(dX_col, x_shape, field_h=3, field_w=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    k, i, j = get_im2col_indices(x_shape, field_h, field_w, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]