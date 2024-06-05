import numpy as np
from activation import *

class dense:
    '''
    Class that contains the implementation of a dense layer

    Attributes:
        units (int): Number of units in the dense operation
        activation (str): Activation applied after the operation
        weight_params (np.ndarray): Trainable weight parameters
        bias_params (np.ndarray): Trainable bias parameters
        output (np.ndarray): Output array of completed dense operation
    '''

    def __init__(self, units: int, activation: str = None) -> None:
        # Initalises all base parameters for the dense layer object
        self.units = units
        self.activation = activation_func(activation)

    def init_params(self, input_size: tuple) -> None:
        # Initialises the weights and biases for the network
        input_neurons = np.prod(input_size[1:])
        self.weight_params = np.random.randn(self.units, input_neurons) * np.sqrt(2 / (input_neurons + self.units))
        self.bias_params = np.zeros(self.units)

        self.output = self.calculate_size(input_size)

    def calculate_size(self, input_size: tuple) -> None:
        # Calculates the expected output size
        return np.zeros([input_size[0], self.units])

    def __call__(self, input: np.ndarray) -> np.ndarray:
        # Performs the dense operation
        self.input = input
        self.output = np.dot(input, self.weight_params.T) + self.bias_params
        return self.activation(self.output)
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Backwards function which updates the weight and bias parameters, calculates the input gradient for the next layer
        output_gradient = self.activation.backward(output_gradient)

        weights_gradient = np.dot(output_gradient.T, self.input)
        input_gradient = np.dot(output_gradient, self.weight_params)

        self.weight_params -= learning_rate * (weights_gradient / self.input.shape[0])
        self.bias_params -= learning_rate * np.mean(output_gradient, axis=0)

        return input_gradient