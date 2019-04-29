import numpy as np
from activations import Activation, DerivativeOfActivation
from loss import Loss


class Neural_network:
    def __init__(self, x_shape, y_shape):
        self.input_shape = x_shape
        self.weights_1 = np.random.rand(self.input_shape[1], 4)
        self.weights_2 = np.random.rand(self.weights_1.shape[1], y_shape[1])
        self.y_shape = y_shape

        self.layer_1_output = None
        self.layer_2_output = None
        self.output = None

    def feedforward(self, x):
        self.layer_1_output = Activation.sigmoid(np.dot(x, self.weights_1))
        self.layer_2_output = Activation.sigmoid(
            np.dot(self.layer_1_output, self.weights_2))

        self.output = self.layer_2_output
        return self.output

    def backpropagation(self, x, y):
        delta_weights_2 = np.dot(
            self.layer_1_output, 2*(y - self.output)*DerivativeOfActivation.sigmoid(self.output))
        delta_weights_1 = np.dot(x.T, np.dot(2*(y - self.output)*DerivativeOfActivation.sigmoid(
            self.output), self.weights_2.T)*DerivativeOfActivation.sigmoid(self.layer_1_output))
        self.weights_1 += delta_weights_1
        self.weights_2 += delta_weights_2

    def train(self, x, y):
        self.feedforward(x)
        self.backpropagation(x, y)



X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float)
y = np.array(([0], [1], [1], [0]), dtype=float)

nn = Neural_network(x_shape=(None, 2), y_shape=(None, 1))
for i in range(1500):
    if i % 100:
        print("Iteration {}".format(i))
        print("Input {}".format(str(X)))
        print("Actual output {}".format(str(y)))
        print("Predicted output {}".format(nn.feedforward(X)))
        print("Loss MSE: {}".format(Loss.mse(nn.feedforward(X), y)))
        print("------\n")
    nn.train(X, y)


