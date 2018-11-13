#-*- coding:utf-8 -*-

import numpy as np


################
# np_pca
################
def np_pca(X, k):
    # mean of each feature
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    norm_X = X - mean

    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)

    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]

    eig_pairs.sort(reverse=True)
    feature = np.array([ele[1] for ele in eig_pairs[:k]])

    data = np.dot(norm_X, np.transpose(feature))
    return data



X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#print(np_pca(X,1))

################
# sklearn PCA
################
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
pca.fit(X)
newX = pca.transform(X)
#print(newX)


################
# fancy PCA
################
def fancyPCA(img_path):
    im = cv2.imread(img_path)
    img = im / 255.

    result = np.zeros(shape=img.shape)

    flag = True
    while flag:
        m = len(img)
        #print(m)
        n = len(img[0])
        #print(n)
        im1 = img[:,:,0].reshape([1, m*n])
        im2 = img[:,:,1].reshape([1, m*n])
        im3 = img[:,:,2].reshape([1, m*n])

        d1 = im1 - np.mean(im1)
        d2 = im2 - np.mean(im2)
        d3 = im3 - np.mean(im3)

        data = np.hstack((d1.T, d2.T, d3.T))
        datat = data.T

        val, vec = np.linalg.eig(np.dot(datat, data))

        dval = np.sqrt(val).reshape([3, 1])

        alpha = 0.1 * np.random.rand(3, 1)

        noise = np.dot(vec, np.multiply(alpha, dval))
        #print(noise.shape)

        b, g, r = cv2.split(img)
        b = b + 4.5 *(noise[0]/255.)
        g = g + 4.5 *(noise[1]/255.)
        r = r + 4.5 *(noise[2]/255.)

        result = cv2.merge((b, g, r))

        lowerThreshold = 0.0
        upperThreshold = 1.0
        if (np.mean(result) < lowerThreshold) | (np.mean(result) > upperThreshold):
            print(np.mean(result))
        else:
            flag = False
            return result

    return result

import cv2
img = fancyPCA('img_1.jpg')

cv2.imshow('img_1', img*255)
cv2.waitKey()


