import numpy as np

def activation_func(activation):
    # Maps the user entered activation function to the written class
    activation_map = {
        None: NoActivation(),
        'relu': reLU(),
        'sigmoid': sigmoid(),
        'softmax': softmax(),
    }

    return activation_map[activation]

class NoActivation:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        # Returns the input if no activation is selected
        return input
    
    def backward(self, output: np.ndarray) -> np.ndarray:
        # Returns the input if no activation is selected
        return output

class reLU:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        # Applies the reLU activation function to the input tensor
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, output: np.ndarray) -> np.ndarray:
        # Calculates the derivative of the reLU function and multiplies it by the gradient
        return np.multiply(np.where(self.input > 0, 1, 0), output)
    
class sigmoid:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        # Applies the sigmoid activation function to the input tensor
        self.input = 1 / (1 + np.exp(-input))
        return self.input
    
    def backward(self, output: np.ndarray) -> np.ndarray:
        # Calculates the derivative of the sigmoid function and multiplies it by the gradient
        return np.multiply(self.input * (1 - self.input), output)
    
class softmax:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, output: np.ndarray) -> np.ndarray:
        raise NotImplementedError