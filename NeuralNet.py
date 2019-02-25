# Jacob Lewis
# Creating a back-prop NN in python
# referenced https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6 for
# some assistance

import numpy as np
import math
from sklearn import datasets

class NeuralNetwork:
    """
    TODO: Class Docstring
    """
    def __init__(self, x, y, hidden_layer_sizes=(100,)):
        """
        Constructor
        :param x: input
        :param y: output
        :param hidden_layer_sizes:
        """
        self.x = x
        self.y = y
        self.layers = []
        self.n_layers = len(hidden_layer_sizes)
        self.weights = []
        self.weights.append(np.random.uniform(-1/(math.sqrt(hidden_layer_sizes[0])), 1/(math.sqrt(hidden_layer_sizes[0])),
                                         range()))

        print(str(self.n_layers))
        

    def feedforward(self):
        """
        Performs the feedforward action in the NN
        :return:
        """
        # create the first layer using the NN input x
        first_layer = relu(np.dot(self.x, self.weights))
        self.layers.append(first_layer)

        layer = None
        for i in range(self.n_layers-1):
            layer = relu(np.dot(self.layers[i], self.weights))

        return layer

    def create_weights(self):
        """
        Create the weights for each layer in the NN
        """




def relu(x):
    """
    Create the relu activation function. Decided to use the definitive relu
    :param x: numpy array of values
    """
    return x * (x > 0)


def relu_derivative(x):
    """
    Create the relu derivative activation function
    :param x: numpy array of values
    """
    return 1. * (x > 0)




def main():
    digits = datasets.load_digits()
    images_and_labels = list(zip(digits.images, digits.target))
    NeuralNetwork(digits.data, digits.target)


if __name__ == "__main__":
    main()