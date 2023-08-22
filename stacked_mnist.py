import tensorflow as tf
import numpy as np
import os

NUM_TRAIN = 100000
NUM_TEST = 25000

def get_1d_label(lbl):
    new_label = 0
    for i in range(3):
        new_label += lbl[i]*10**(2-i)
    return int(new_label)

def load_data(padding = False):

    (train_img, train_label), (test_img, test_label) = tf.keras.datasets.mnist.load_data()
    stacked_mnist_train_img = np.zeros((NUM_TRAIN, 32, 32, 3))
    stacked_mnist_train_label = np.zeros(NUM_TRAIN, dtype='int')
    stacked_mnist_test_img = np.zeros((NUM_TEST, 32, 32, 3))
    stacked_mnist_test_label = np.zeros(NUM_TEST, dtype = 'int')

    for i in range(NUM_TRAIN):
        idx = np.random.randint(0, train_img.shape[0], size = 3)
        idx = np.random.permutation(idx)
        image = np.dstack(train_img[idx])
        if padding:
            image = np.pad(image, ((2,2), (2, 2), (0, 0)))
        label_3d = train_label[idx]
        label = get_1d_label(label_3d)
        stacked_mnist_train_img[i] = image
        stacked_mnist_train_label[i] = label
        
    for i in range(NUM_TEST):
        idx = np.random.randint(0, test_img.shape[0], size = 3)
        idx = np.random.permutation(idx)
        image = np.dstack(test_img[idx])
        if padding:
            image = np.pad(image, ((2,2), (2, 2), (0, 0)))
        label_3d = test_label[idx]
        label = get_1d_label(label_3d)
        stacked_mnist_test_img[i] = image
        stacked_mnist_test_label[i] = label

    return (stacked_mnist_train_img, stacked_mnist_train_label), (stacked_mnist_test_img, stacked_mnist_test_label)

