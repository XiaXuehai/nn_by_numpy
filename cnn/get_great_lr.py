# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from read_mnist import *
from time import *

batch_size = 64


class xnet(nn.Module):
    def __init__(self):
        super(xnet, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 3, 1, 1)
        self.conv2 = nn.Conv2d(10, 10, 3, 1, 1)
        self.fc1 = nn.Linear(490, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 490)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def get_minibatch(x, y, minibatch_size):
    minibatches = []
    # get random index for shuffle
    r = np.random.permutation(len(x))
    for i in range(0, x.shape[0], minibatch_size):
        idx = r[i:i + minibatch_size]
        x_mini = x[idx]
        y_mini = y[idx]
        minibatches.append((x_mini, y_mini))
    return minibatches


x_train = get_image('../mnist_data/raw/train-images-idx3-ubyte')
y_train = get_label('../mnist_data/raw/train-labels-idx1-ubyte')
x_test = get_image('../mnist_data/raw/t10k-images-idx3-ubyte')
y_test = get_label('../mnist_data/raw/t10k-labels-idx1-ubyte')
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()

minibatches = get_minibatch(x_train, y_train, batch_size)
epoches = 3
net = xnet()

lr_mult = (1 / 1e-5) ** (1 / 100)
lrs = []
losses = []
best_loss = 1e9
optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_mult)

for epoch in range(epoches):
    for i in range(len(minibatches)):

        x, y = minibatches[i]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()

        out = net(x)
        loss = F.cross_entropy(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        lrs.append(lr)
        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > 4 * best_loss or lr > 1.:
            break

    #     if (i+1)% 100==0:
    #         print('step:{}, loss:{:.2}'.format(i+1, loss.item()))
    #
    # with torch.no_grad():
    #     test_out = net(x_test)
    #     test_out = F.softmax(test_out, dim=1)
    #     y_pred = torch.argmax(test_out, dim=1)
    #     print((y_pred==y_test).sum().item()/ 10000)

plt.figure()
plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
plt.xlabel('learning rate')
plt.ylabel('loss')
plt.plot(np.log(lrs), losses)
plt.show()
# plt.figure()
# plt.xlabel('num iterations')
# plt.ylabel('learning rate')
# plt.plot(lrs)
# plt.show()






