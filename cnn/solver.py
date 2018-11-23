# coding:utf-8

import numpy as np


def get_minibatch(x, y, minibatch_size, shuffle=True):
    minibatches=[]

    # get random index for shuffle
    r = np.random.permutation(len(x))

    for i in range(0, x.shape[0], minibatch_size):
        idx = r[i:i+minibatch_size]
        x_mini = x[idx]
        y_mini = y[idx]
        minibatches.append((x_mini, y_mini))

    return minibatches


def sgd(nn, x_train, y_train, val_set=None, lr=1e-3, mb_size=256, n_iter=2000, print_after=100):
    minibatches = get_minibatch(x_train, y_train, mb_size)

    if val_set:
        x_val, y_val = val_set

    for iter in range(1, n_iter+1):
        # get minibatches randomly
        idx = np.random.randint(0, len(minibatches))
        x_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(x_mini, y_mini)
        if iter % print_after == 0:
            if val_set:
                pass #TODO
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for layer in grad:
            nn.model[layer] -= lr * grad[layer]

    return nn


if __name__ == '__main__':
    pass







