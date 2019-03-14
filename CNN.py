# Jacob Lewis
# CNN from scratch

import tensorflow as tf
import numpy as np

class CNN:
    def __init__(self):
        pass


def get_data():
    # created using code from https://github.com/wssrcok/QNN.np/blob/master/utils.py
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    classes = 10
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_labels = one_hot_label(classes, train_labels)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels_old = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_labels = one_hot_label(classes, eval_labels_old)

    train_data = train_data.reshape(55000, 1, 28, 28)
    eval_data = eval_data.reshape(10000, 1, 28, 28)

    return train_data, train_labels, eval_data, eval_labels, classes

def one_hot_label(classes, label):
    # created using code from https://github.com/wssrcok/QNN.np/blob/master/utils.py
    """
    reshape label to Sam prefered shape for mnist
    Arguments:
    label -- input label with shape (m,)
    Returns:
    new_label -- output label with shape (classes, m)
    """
    m = label.shape[0]
    new_label = np.zeros((classes, m))
    for i in range(m):
        clas = label[i]
        new_label[clas,i] = 1
    return new_label

def main():
    train_data, train_labels, test_data, test_labels, classes = get_data()
    pass


if __name__ == '__main__':
    main()
