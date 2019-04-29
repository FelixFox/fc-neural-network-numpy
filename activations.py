import numpy as np


class Activation:
    def sigmoid(t):
        return 1/(1+np.exp(-t))


class DerivativeOfActivation:
    def sigmoid(t):
        return t*(1-t)
