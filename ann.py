###
#http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
###
import numpy as np
from sklearn.datasets import *
import matplotlib.pyplot as plt
import copy

np.random.seed(0)
X, y = make_moons(200, noise=0.20)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()

num_input = X.shape[0]
nn_input_dim = X.shape[1]
nn_output_dim= 2

lr = 0.001
lamda = 0.01

def predict(model, X):
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    z1 = X.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) +b2
    z2_softmax = np.exp(z2)
    probs = z2_softmax/np.sum(z2_softmax, axis=1, keepdims=True)

    return np.argmax(probs, axis=1)

def build_model(nn_hidden_dim, X, epoch=2000):
    np.random.seed(0)
    w1 = np.random.randn(nn_input_dim, nn_hidden_dim)/np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hidden_dim))
    w2 = np.random.randn(nn_hidden_dim, nn_output_dim)/np.sqrt(nn_hidden_dim)
    b2 = np.zeros((1, nn_output_dim))

    data_loss = 0.0
    model = {}

    for i in range(epoch):

        z1 = X.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        z2_softmax = np.exp(z2)
        probs = z2_softmax/np.sum(z2_softmax, axis=1, keepdims=True)

        delta3 = copy.deepcopy(probs)
        delta3[range(num_input), y] -= 1
        dw2 = np.dot(a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(w2.T) * (1 - np.power(a1, 2))
        dw1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        dw2 += lamda * w2
        dw1 += lamda * w1

        w1 += -lr * dw1
        b1 += -lr * db1
        w2 += -lr * dw2
        b2 += -lr * db2

        loss = -np.log(probs[range(num_input), y]) #got cross-entropy
        loss = np.sum(loss)
        loss += lamda/2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
        loss = 1./num_input * loss


        model = {'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}

        if i%1000 == 0:
            print('Step:{}, Loss:{:.4f}'.format(i, loss))

    return model

model = build_model(3, X)

pred = predict(model, X)
print('acc = ',np.sum(pred==y)/200.)

