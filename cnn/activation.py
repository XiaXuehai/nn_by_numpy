# coding:utf-8

import numpy as np

def fc_forward(X, W, b):
    out = X @ W + b # np.matmul(x, w)
    cache = (W, X)
    return out, cache

# https://blog.csdn.net/l691899397/article/details/52267166
def fc_backward(dout, cache):
    w, h = cache
    # remember it !!!
    dW = h.T @ dout
    db = np.sum(dout, axis=0)
    dX = dout @ w.T

    return dX, dW, db


def relu_forward(x):
    out = np.maximum(x, 0) # compare with 0
    cache = x
    return out, cache

def relu_backward(dout, cache):
    dX = dout.copy()
    dX[cache<=0] = 0
    return dX

def sigmoid_forward(x):
    out = 1 / (1 + np.exp(-x))
    cache = out
    return out, cache

def sigmoid_backward(dout, cache):
    dx = cache * (1 - cache) * dout
    return dx

def tanh_forward(x):
    out = np.tanh(x)
    cache = out
    return out, cache

def tanh_backward(dout, cache):
    dx = (1 - cache**2) * dout
    return dx

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


# https://blog.csdn.net/yuechuen/article/details/71502503
# 每个bn都有四个参数，mean,var, gamma,beta
# 其中 gamma和beta是需要训练的参数
# bn作用于非线性映射之前，防止梯度弥散
# 训练时：1.收敛速度很慢，2.梯度爆炸等无法训练状况
#        3. 在一般使用情况下也可以加入BN来加快训练速度，提高模型精度。
def bn_forward(x, gamma, beta, cache, momentum=0.9, train=True):
    running_mean, running_var = cache

    if train:
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        x_norm = (x - mu) / np.sqrt(var + 1e-7)
        # gamma and beta are the parameters to be learned
        out = gamma * x_norm + beta

        cache = (x, x_norm, mu, var, gamma, beta)
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var

    else:
        x_norm = (x - running_mean) / np.sqrt(running_var + 1e-8)
        out = gamma * x_norm + beta
        cache = None

    return out, cache, running_mean, running_var

def bn_backward(dout, cache):
    x, x_norm, mu, var, gamma, beta = cache

    N, D = x.shape
    x_mu = x - mu
    std_inv = 1. / np.sqrt(var + 1e-7)

    dx_norm = dout * gamma
    dvar = -0.5 * np.sum(dx_norm * x_mu, axis=0)*std_inv**3
    dmu = -std_inv * np.sum(dx_norm, axis=0) - 2 *dvar * np.sum(x_mu, axis=0) / N

    dx = (dx_norm * std_inv) + (dvar * 2 * x_mu / N) + (dmu / N)
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dx, dgamma, dbeta


def dropout_forward(x, p_dropout):
    u = np.random.binomial(1, p_dropout, size=x.shape)
    u /= p_dropout # rescale the output,correct the expectation of output
    out = x * u
    cache = u
    return out, cache

def dropout_backward(dout, cache):
    dx = dout * cache
    return dx

if __name__ == '__main__':
    pass


