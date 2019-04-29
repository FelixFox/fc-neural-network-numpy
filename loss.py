import numpy as np 


class Loss:
    def mse(predicted, actual):
        return np.mean(np.square(actual - predicted))
