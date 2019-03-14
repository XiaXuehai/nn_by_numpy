# coding:utf-8
# https://www.cnblogs.com/callyblog/p/8299074.html
# sgd 依赖初始化，且跑不出鞍点
# 数据是稀疏的，要快速收敛，网络复杂，用adaptive的方法，不用fine-tuning lr
# rmsprop 是adaggrad的改进版本，解决lr减小过快
# Adam=偏移修正+momentum+rmsprop， 稀疏梯度中，偏移修正使得adam在最终收敛快于rmsprop
# RMSprop, Adadelta, Adam 是属一类，adam是最好选择
# 最近论文中用：SGD
# 所有的优化，都是让单个或少批量样本不影响总样本的趋势，用二阶动量去度量历史更新频率
import numpy as np


def get_minibatch(x, y, minibatch_size):
    minibatches=[]
    # get random index for shuffle
    r = np.random.permutation(len(x))
    for i in range(0, x.shape[0], minibatch_size):
        idx = r[i:i+minibatch_size]
        x_mini = x[idx]
        y_mini = y[idx]
        minibatches.append((x_mini, y_mini))
    return minibatches

def sgd(nn, x_train, y_train, val_set=None, lr=1e-3, mb_size=256, print_after=100):
    minibatches = get_minibatch(x_train, y_train, mb_size)

    if val_set:
        x_val, y_val = val_set

    for iter in range(len(minibatches)):
        x_mini, y_mini = minibatches[iter]
        grad, loss = nn.train_step(x_mini, y_mini)
        if iter % print_after == 0:
            if val_set:
                pass #TODO
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for layer in grad:
            nn.model[layer] -= lr * grad[layer]
    return nn

# To prevent shock because of large learning rate
# nesterov momentum is copy nn, and update the w,b, and train_step and then,then...
def momentum(nn, x_train, y_train, val_set=None, lr=1e-3, mb_size=256, print_after=100):
    gamma = 0.9
    velocity = {k: np.zeros_like(v) for k, v in nn.model.items()}
    minibatches = get_minibatch(x_train, y_train, mb_size)

    if val_set:
        x_val, y_val = val_set

    for iter in range(len(minibatches)):
        x_mini, y_mini = minibatches[iter]
        grad, loss = nn.train_step(x_mini, y_mini)
        if iter % print_after == 0:
            if val_set:
                pass #TODO
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for layer in grad:
            velocity[layer] = lr * grad[layer] + gamma * velocity[layer]
            nn.model[layer] -= velocity[layer]
    return nn

# adagrad 在训练中调整学习率；适合稀疏样本，小梯度大学习率
# 在一个epoch的训练后期，cache越来越大，导致分母越来越大，是的更新梯度为0，提前结束
# 这个公式很好：https://www.jianshu.com/p/a8637d1bb3fc
def adagrad(nn, x_train, y_train, val_set=None, lr=1e-3, mb_size=256, print_after=100):
    cache = {k: np.zeros_like(v) for k, v in nn.model.items()}
    minibatches = get_minibatch(x_train, y_train, mb_size)

    if val_set:
        x_val, y_val = val_set

    for iter in range(len(minibatches)):
        x_mini, y_mini = minibatches[iter]

        grad, loss = nn.train_step(x_mini, y_mini)
        if iter % print_after == 0:
            if val_set:
                pass  # TODO
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for layer in grad:
            cache[layer] += grad[layer]**2
            nn.model[layer] -= lr/(np.sqrt(cache[layer])+1e-8) * grad[layer]

    return nn

# 累加之前的梯度平方，但是加入了一个系数，类似于加权平均数
# 使得幅度变小，避免二阶动量持续累加，导致训练提前结束
def rmsprop(nn, x_train, y_train, val_set=None, lr=1e-3, mb_size=256, print_after=100):
    gamma = 0.9
    cache = {k: np.zeros_like(v) for k, v in nn.model.items()}
    minibatches = get_minibatch(x_train, y_train, mb_size)

    if val_set:
        x_val, y_val = val_set

    for iter in range(len(minibatches)):
        x_mini, y_mini = minibatches[iter]

        grad, loss = nn.train_step(x_mini, y_mini)
        if iter % print_after == 0:
            if val_set:
                pass  # TODO
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for layer in grad:
            cache[layer] = gamma * cache[layer] + (1 - gamma) * grad[layer]**2
            nn.model[layer] -= lr/(np.sqrt(cache[layer])+1e-7) * grad[layer]

    return nn


def adam(nn, x_train, y_train, val_set=None, lr=1e-3, mb_size=256, print_after=100):
    beta1 = 0.9
    beta2 = 0.999
    M = {k: np.zeros_like(v) for k, v in nn.model.items()}
    R = {k: np.zeros_like(v) for k, v in nn.model.items()}
    minibatches = get_minibatch(x_train, y_train, mb_size)

    if val_set:
        x_val, y_val = val_set

    for iter in range(len(minibatches)):
        t = iter + 1
        x_mini, y_mini = minibatches[iter]

        grad, loss = nn.train_step(x_mini, y_mini)
        if iter % print_after == 0:
            if val_set:
                pass  # TODO
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for layer in grad:
            M[layer] = beta1 * M[layer] + (1 - beta1) * grad[layer]
            R[layer] = beta2 * R[layer] + (1 - beta2) * (grad[layer]**2)

            _M = M[layer] / (1 - beta1**t)
            _R = R[layer] / (1 - beta2**t)

            nn.model[layer] -= lr * _M / (np.sqrt(_R)+1e-7)

    return nn






