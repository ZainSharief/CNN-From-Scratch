import numpy as np

class flatten:
    '''
    Class that contains the implementation of a flatten layer

    Attributes:
        input_shape (np.ndarray): Shape inputted to the flatten operation
    '''

    def __init__(self):
        self.input_shape = None

    def __call__(self, input: np.ndarray) -> np.ndarray:
        # Performs the flatten operation
        self.input_shape = input.shape
        return np.reshape(input, (input.shape[0], -1))

    def backward(self, output_gradient: np.ndarray, _: float) -> np.ndarray:
        # Backwards function which returns the input to its original shape
        return np.reshape(output_gradient, self.input_shape)