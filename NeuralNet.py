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
        self.y = y/10.
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = 1
        self.layers = []
        self.n_layers = len(hidden_layer_sizes)
        self.weights = []


        self.output = self.feedforward()
        self.backprop()
        

    def feedforward(self):
        """
        Performs the feedforward action in the NN
        :return:
        """
        # get the initial weights for each layer
        self.create_weights()

        # create the first layer using the NN input x
        first_layer = relu(np.dot(self.x, self.weights[0]))
        self.layers.append(first_layer)

        layer = None
        for i in range(self.n_layers-1):
            layer = relu(np.dot(self.layers[i], self.weights[i+1]))
            self.layers.append(layer)

        return layer

    def backprop(self):
        """
        Performs the back-propagation for the neural network
        :return:
        """
        print(len(self.weights))
        print(len(self.layers))


        adjustments = []
        output_error = self.y - self.output
        output_delta = np.dot(output_error, relu_derivative(self.output))
        output_correction = self.output(1-self.output)*output_error

        for index, layer in reversed(list(enumerate(self.layers))):
            error = np.dot(output_delta, self.weights[index])
            delta = error*relu_derivative(layer)
            adjustment = 0



    def create_weights(self):
        """
        Create the weights for each layer in the NN
        """
        for i in range(self.n_layers):
            in_size = self.hidden_layer_sizes[i]
            out_size = self.output_size
            if len(self.hidden_layer_sizes) > i+1:
                out_size = self.hidden_layer_sizes[i+1]
            self.weights.append(np.random.uniform(-1/(math.sqrt(self.hidden_layer_sizes[i])),
                                                  1/(math.sqrt(self.hidden_layer_sizes[i])),
                                                  (in_size, out_size)))




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
    NeuralNetwork(digits.data, digits.target, hidden_layer_sizes=(64, 32))


if __name__ == "__main__":
    main()