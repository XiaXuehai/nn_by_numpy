# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


def read_mnist():
    im = get_image()
    label = get_label()
    return im, label


def get_image():
    with open('mnist_data/train-images-idx3-ubyte','rb') as f1:
        buf1 = f1.read()

    image_size = 28
    image_index = 16 # the first 16bytes is four I type
    img = np.ndarray(len(buf1) - image_index, '>B', buf1, image_index)
    im = img.reshape(-1, 1,image_size, image_size)

    return im


def get_label():
    with open('mnist_data/train-labels-idx1-ubyte','rb') as f2:
        buf2 = f2.read()

    label_index = 8 # the first 8bytes is two I type
    label = np.ndarray(len(buf2) - label_index, '>B', buf2, label_index)

    return label


if __name__ == "__main__":
    im, label = read_mnist()

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        title = 'label:'+ str(label[i])
        plt.title(title)
        plt.imshow(im[i], cmap='gray')
    plt.show()

