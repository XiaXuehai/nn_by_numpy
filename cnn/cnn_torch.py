# coding:utf-8

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from read_mnist import *

batch_size = 50


class xnet(nn.Module):
    def __init__(self):
        super(xnet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16, momentum=0.9, eps=1e-6),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32, momentum=0.9, eps=1e-6),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.tanh(out)
        out = self.layer2(out)
        out = torch.tanh(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


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
if torch.cuda.is_available():
    x_test = x_test.cuda()
    y_test = y_test.cuda()

minibatches = get_minibatch(x_train, y_train, batch_size)
epoches = 50
if torch.cuda.is_available():
    net = xnet().cuda()
else:
    net = xnet()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

for epoch in range(epoches):
    for i in range(len(minibatches)):
        x, y = minibatches[i]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        out = net(x)
        loss = F.cross_entropy(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch:{},step:{}, loss:{:.2}'.format(epoch, i + 1, loss.item()))

    with torch.no_grad():
        test_out = net(x_test)
        test_out = F.softmax(test_out, dim=1)
        y_pred = torch.argmax(test_out, dim=1)
        print((y_pred == y_test).sum().item() / 10000)

torch.save(net.state_dict(), 'cnn_torch.pt')






