# coding:utf-8

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
        # get minibatches randomly
        # idx = np.random.randint(0, len(minibatches))
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

def adagrad(nn, x_train, y_train, val_set=None, lr=1e-3, mb_size=256, print_after=100):
    cache = {k: np.zeros_like(v) for k, v in nn.model.items()}
    minibatches = get_minibatch(x_train, y_train, mb_size)

    if val_set:
        x_val, y_val = val_set

    for iter in range(len(minibatches)):
        # get minibatches randomly
        # idx = np.random.randint(0, len(minibatches))
        x_mini, y_mini = minibatches[iter]

        grad, loss = nn.train_step(x_mini, y_mini)
        if iter % print_after == 0:
            if val_set:
                pass  # TODO
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for layer in grad:
            cache[layer] += grad[layer]**2
            nn.model[layer] -= lr/(np.sqrt(cache[layer])+1e-7) * grad[layer]

    return nn

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

if __name__ == '__main__':
    pass







