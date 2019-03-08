# Jacob Lewis
# Creating a back-prop NN in python
# referenced https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6 for
# some assistance
# also referenced https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
# also referenced https://github.com/Alescontrela/Numpy-CNN/tree/master/CNN
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
        self.y = y
        # self.y = np.array(y/10.).reshape(len(y), 1)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = 1

        # create the filters for
        # hardcoded, needs to be updated
        self.filter = np.zeros((2, 3, 3))
        self.filter = np.random.normal(size=(8, 1, 5, 5))
        # self.filter[1, :, :] = np.array([[[1, 1, 1],
        #                              [0, 0, 0],
        #                              [-1, -1, -1]]])

        # create new tuple of every layer size
        self.layer_sizes = (len(self.x[1]),) + self.hidden_layer_sizes + (self.output_size,)
        self.layers = []
        self.n_layers = len(self.layer_sizes)
        # self.weights = []
        self.weights = [np.random.standard_normal(size=(128, 128)) * 0.01, np.random.standard_normal(size=(10, 128)) * 0.01]
        # self.output = np.zeros(y.shape)
        # get the initial weights for each layer
        # self.create_weights()
        self.run()

    def run(self):
        for i in range(1):
            loss = self.train()
            if i % 100 is 0:
                print("yay")
            print(str(loss))

    def train(self):
        losses = []
        for index, image in enumerate(self.x):
            grads, loss = self.conv(image.reshape((1, 8, 8)), self.y[index], self.filter, self.weights[0], self.weights[1])

            print(self.y[index])

            self.filter += grads[0]
            self.weights[0] += grads[1]
            self.weights[1] += grads[2]

            losses.append(loss)

        print(losses)

        return np.mean(losses)

    def conv(self, image, label, filter, weight1, weight2):
        conved = self.convolution(image, filter)
        conved = relu(conved)

        fc = conved.reshape((conved.shape[0] * conved.shape[1] * conved.shape[1], 1))

        z = relu(weight1.dot(fc))

        probabilities = self.FC(weight2.dot(z))

        loss = -np.sum(label * np.log(probabilities))

        dout = probabilities - label
        dweight2 = dout.dot(z.T)
        dz = relu_derivative(weight2.T.dot(dout))

        dweight1 = dz.dot(fc.T)

        dfc = weight1.T.dot(dz)

        dconved = relu_derivative(dfc.reshape(conved.shape))

        dimage, dfilter = self.convolutionBackward(dconved, image, filter)

        return (dfilter, dweight1, dweight2), loss




    def convolution(self, image, filter, s=1):
        # created with help from https://github.com/Alescontrela/Numpy-CNN/blob/master/CNN/forward.py
        out_dim = int(image.shape[1] - filter.shape[2]) + 1
        output = np.zeros((filter.shape[0], out_dim, out_dim))

        assert image.shape[0] == filter.shape[1]


        # perform the convolution
        for filter_num in range(filter.shape[0]):
            y = 0
            output_y = 0
            while y + filter.shape[2] <= image.shape[1]:
                x = 0
                output_x = 0
                while x + filter.shape[2] <= image.shape[1]:
                    # print(image[y:y+filter.shape[2], x:x+filter.shape[2]])
                    output[filter_num, output_y, output_x] = np.sum(filter[filter_num] * image[:, y:y+filter.shape[2], x:x+filter.shape[2]])
                    x += s
                    output_x += 1
                y += s
                output_y += 1

        return output

    def convolutionBackward(self, prev, inp, filter, s=1):
        # created with help from https://github.com/Alescontrela/Numpy-CNN/blob/master/CNN/backward.py
        dout = np.zeros(inp.shape)
        dfilt = np.zeros(filter.shape)

        for filter_num in range(filter.shape[0]):
            y = 0
            output_y = 0
            while y + filter.shape[2] <= inp.shape[1]:
                x = 0
                output_x = 0
                while x + filter.shape[2] <= inp.shape[1]:
                    dfilt[filter_num] += prev[filter_num, output_y, output_x] * inp[:, y:y+filter.shape[2], x:x+filter.shape[2]]
                    dout[:, y:y+filter.shape[2], x:x+filter.shape[2]] = prev[filter_num, output_y, output_x] * filter[filter_num]
                    x += s
                    output_x += 1
                y += s
                output_y += 1

        return dout, dfilt


    def FC(self, X):
        out = np.exp(X)
        return out / np.sum(out)

    # def conv(self, img, conv_filter):
    #     feature_maps = np.zeros((img.shape[0] - conv_filter.shape[1] + 1, img.shape[1]-conv_filter.shape[1]+1, conv_filter.shape[0]))
    #     for filter_num in range(conv_filter.shape[0]):
    #         curr_filter = conv_filter[filter_num, :]
    #         conv_map = self.conv_(img[:, :, 0], curr_filter[:, :, 0])
    #         for ch_num in range(1, curr_filter.shape[-1]):
    #             conv_map = conv_map + self.conv_(img[:, :, ch_num],
    #                                         curr_filter[:, :, ch_num])
    #     feature_maps[:, :, filter_num] = conv_map
    #
    # def conv_(self, img, conv_filter):
    #     result = np.zeros(img.shape)
    #     for r in np.uint16(np.arange(filter))
    #     return final_result


    def feedforward(self):
        """
        Performs the feedforward action in the NN
        :return:
        """
        self.layers = []
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

        delta = output_error * relu_derivative(self.output)
        # output_delta = np.dot(output_error, relu_derivative(self.output))
        adjustments = [np.dot(self.layers[len(self.layers)-1].T, delta)]

        # remove the output layer from our layers variable
        self.layers.pop()

        for index, layer in reversed(list(enumerate(self.layers))):
            x = self.weights[index+1].T
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
            # print("Adjustment: " + str(adjustment))
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

    # def train(self):
    #     self.output = self.feedforward()
    #     self.backprop()

    # def run(self):
    #     for i in range(1500):  # trains the NN 1,500 times
    #         self.train()
    #         if i % 100 == 0:
    #             print("for iteration # " + str(i))
    #             # print ("Input : \n" + str(self.x))
    #             # print ("Actual Output: \n" + str(self.y))
    #             # print ("Predicted Output: \n" + str(self.feedforward()))
    #             print("Loss: \n" + str(np.mean(np.square(self.y - self.output))))  # mean sum squared loss
    #             # print("Weight 0: " + str(self.weights[0]))
    #             print("\n")




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
    images = datasets.load_digits()
    # images_and_labels = list(zip(diabetes.images, diabetes.target))
    # mynn = NeuralNetwork(images.data, images.target, hidden_layer_sizes=(1000,))
    mynn = NeuralNetwork(images.data, images.target)

    # print(digits.images[0])
    # mynn.run()






if __name__ == "__main__":
    main()