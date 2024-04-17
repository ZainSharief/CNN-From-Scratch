import numpy as np

class flatten:
    def __init__(self):
        self.input_shape = None

    def __call__(self, input):
        # Performs the flatten operation
        self.input_shape = input.shape
        return np.reshape(input, (input.shape[0], -1))

    def backward(self, output_gradient, _):
        # Backwards function which returns the input to its original shape
        return np.reshape(output_gradient, self.input_shape)