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
        :param hidden_layer_sizes: tuple of the sizes of the hidden layers
        """
        self.x = x
        self.y = np.array(y/10.).reshape(len(y), 1)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = 1
        # create new tuple of every layer size
        self.layer_sizes = (len(self.x[1]),) + self.hidden_layer_sizes + (self.output_size,)
        self.layers = []
        self.n_layers = len(self.layer_sizes)
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
        layer = relu(np.dot(self.x, self.weights[0]))
        self.layers.append(layer)

        for i in range(len(self.weights)-1):
            # for each weight in self.weights create a new layer based on the previous layer
            layer = relu(np.dot(layer, self.weights[i+1]))
            self.layers.append(layer)

        # return the last (output) layer
        return layer

    def backprop(self):
        """
        Performs the back-propagation for the neural network
        :return:
        """
        output_error = self.y - self.output
        # delta = np.dot(output_error, relu_derivative(self.output)) #TODO: CHECK IF THIS IS OK AS A DOT
        delta = output_error * relu_derivative(self.output)
        # output_delta = np.dot(output_error, relu_derivative(self.output))
        adjustments = [np.dot(self.layers[len(self.layers)-1].T, delta)]

        self.layers.pop()

        for index, layer in reversed(list(enumerate(self.layers))):
            error = np.dot(delta, self.weights[index+1].T)
            # print("ERROR SHAPE: "+ str(error.shape))
            # print("WEIGHTSHAPE: " + str(self.weights[index+1].shape))
            delta = error * relu_derivative(layer)
            if index is 0:
                # this adds an extra element containing the input to the end of self.layers
                # it allows the self.layers[index-1] access below to access the "-1" element which will be the last
                # element in the list
                self.layers.append(self.x)
            adjustment = np.dot(self.layers[index-1].T, delta)
            adjustments.insert(0, adjustment)

        for index, adjustment in enumerate(adjustments):
            self.weights[index] += adjustment


    def create_weights(self):
        """
        Create the weights for each layer in the NN. If there are 2 hidden layers, 3 sets of weights will be created
        """
        for i in range(self.n_layers-1):
            # get the size of the start layer
            in_size = self.layer_sizes[i]
            # get the size of the next layer
            out_size = self.layer_sizes[i+1]
            # if len(self.hidden_layer_sizes) > i+1:
            #     out_size = self.hidden_layer_sizes[i+1]
            self.weights.append(np.random.uniform(-1/(math.sqrt(self.layer_sizes[i])),
                                                  1/(math.sqrt(self.layer_sizes[i])),
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