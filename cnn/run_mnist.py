# coding:utf-8
import numpy as np
from read_mnist import *
from solver import *
import nn

n_iter = 1000
lr = 1e-3
mb_size = 64
epoch = 5
print_after = 10
loss = 'cross_ent'
nonlin = 'relu'
solver = 'sgd'

def main():

    x_train, y_train = read_mnist()

    M, D, C = x_train.shape[0], x_train.shape[1], y_train.max()+1

    solvers = dict(
        sgd = sgd
    )

    solver_fun = solvers[solver]
    accs = np.zeros(epoch)

    print('\nEcperimenting on {}\n'.format(solver))

    for k in range(epoch):
        print('Epoch-{}'.format(k+1))
        net = nn.ConvNet(10, C, H=128)
        net = solver_fun(
            net, x_train, y_train, val_set=None, mb_size=mb_size, lr=lr,
            n_iter=n_iter, print_after=print_after
        )

    # y_pred = net.predict(x_test)
    # acc[k] = np.mean(y_pred, y_test)


if __name__ == '__main__':
    main()







