# coding:utf-8
import numpy as np
from read_mnist import *
from solver import *
import nn


lr = 1e-3
mb_size = 64
epoch = 3
print_after = 10
loss = 'cross_ent'
nonlin = 'relu'
solver = 'adam'

def main():

    x_train, y_train = get_image('mnist_data/train-images-idx3-ubyte'), get_label('mnist_data/train-labels-idx1-ubyte')
    x_test, y_test = get_image('mnist_data/t10k-images-idx3-ubyte'), get_label('mnist_data/t10k-labels-idx1-ubyte')

    M, D, C = x_train.shape[0], x_train.shape[1], y_train.max()+1

    solvers = dict(
        sgd = sgd,
        momentum=momentum,
        adagrad=adagrad,
        rmsprop=rmsprop,
        adam=adam
    )

    solver_fun = solvers[solver]
    accs = np.zeros(epoch)

    print('\nExperimenting on {}\n'.format(solver))

    net = nn.ConvNet(10, C, H=128)
    for k in range(epoch):
        print('Epoch-{}'.format(k+1))
        net = solver_fun(
            net, x_train, y_train, val_set=None, mb_size=mb_size, lr=lr,
            print_after=print_after
        )

        y_pred = net.predict(x_test)
        accs[k] = np.mean(y_pred==y_test)
        print('Accuracy: {:.4f}'.format(accs[k]))


if __name__ == '__main__':
    main()







