# coding:utf-8
import numpy as np
import loss as loss_func
import activation as l
import conv
import pooling

import numpy as np

class nn(object):
    loss_funs = dic(
        cross_ent=loss_func.cross_entropy
        )

    dloss_funs = dict(
        cross_entropy=loss_func.dcross_entropy
        )

    forward_nolins = dict(
        relu=l.relu_forward
        )

    backward_nolins = dict(
        relu=l.relu_backward
        )

    def __init__(self, D, C, H, lam=1e-3, p_dropout=0.8, loss='cross_ent', nonlin='relu'):
        self._init_model(D, C, H)

        self.lam = lam
        self.p_dropout = p_dropout
        self.loss = loss # ???
        self.forward_nolin = nn.forward_nolins[nonlin]
        self.backward_nolin = nn.backward_nolins[nonlin]
        self.mode = 'classification'

    def train_step(self, X_train, y_train):
        pass

    def predict_proba(self, X):
        pass

    def predict(self, X):
        pass

    def forward(self, X, train=False):
        raise NotImplementedError()

    def backward(self, y_pred, y_train, cache):
        raise NotImplementedError()

    def _init_model(self, D, C, H):
        raise NotImplementedError()

class ConvNet(nn):
    def __init__(self, D, C, H, lam=1e-3, p_dropout=0.8, loss='cross_ent', nonlin='relu'):
        super(ConvNet, self).__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, train=False):
        # conv-1
        h1, h1_cache = conv.conv_forward(X, self.model['W1'], self.model['b1'])
        h1, nl_cache1 = l.relu_forward(h1)

        # pool-1
        hpool, hpool_cache = pooling.maxpool_forward(h1)
        h2 = hpool.ravel().reshape(X.shape[0], -1)

        # fc-1
        h3, h3_cache = l.fc_forward(h2, self.model['W2'], self.model['b2'])
        h3, nl_cache3 = l.relu_forward(h3)

        # fc-2
        score, score_cache = l.fc_forward(h3, self.model['W3'], self.model['b3'])

        return score, (X, h1_cache, h3_cache, score_cache, hpool_cache, hpool, nl_cache1, nl_cache3)


    def backward(self, y_pred, y_train, cache):
        X, h1_cache, h3_cache, score_cache, hpool_cache, hpool, nl_cache1, nl_cache3 = cache

        # output layer
        grad_y = self.dloss_funs[self.loss](y_pred, y_train)

        # fc-1
        dh3, dw3, db3 = l.fc_backward(grad_y, score_cache)
        dh3 = self.backward_nolin(dh3, nl_cache3)

        dh2, dw2, db2 = l.fc_backward(dh3, h3_cache)
        dh2 = dh2.ravel().reshape(hpool.shape)

        # pool-1
        dpool = pooling.maxpool_backward(dh2, hpool_cache)

        # conv-1
        dh1 = self.backward_nolin(dh2, nl_cache1)
        dx, dw1, db1 = conv.conv_backward(dh1, h1_cache)

        grad = dict(
            W1=dw1, W2=dw2, W3=dw3, b1=db1, b2=db2, b3=db3
        )

        return grad

    def _init_model(self, D, C, H):
        self.model = dict(
            W1=np.random.randn(D, 1, 3, 3) / np.sqrt(D/2),
            W2=np.random.randn(D*14*14, H) / np.sqrt(D*14*14/2),
            W3=np.random.randn(H, C) / np.sqrt(H/2),
            b1=np.zeros((D, 1)),
            b2=np.zeros((1, H)),
            b3=np.zeros((1, C))
        )



if __name__ == '__main__':
    pass







