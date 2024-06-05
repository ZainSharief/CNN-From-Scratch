import numpy as np
import scipy.signal as signal
from activation import *

class conv2d:
    '''
    Class that contains the implementation of a convolutional layer

    Attributes:
        filters (int): Number of filters
        strides (int): Stride of the convolution
        padding (str): Padding type ('valid' or 'same')
        activation (str): Activation function name
        kernel_size (tuple): Size of the kernel (height, width)
        padding_amount (tuple): Amount of padding added to each x and y
        input_size (tuple): Shape of the input
        kernel_params (np.ndarray): Trainable parameters in the kernel
        bias_params (np.ndarray): Trainable bias parameters 
        output (np.ndarray): Output of the convolution
    '''

    def __init__(self, filters: int, kernel_size: tuple = (3,3), strides: int = 1, padding: str = 'valid', activation: str = None) -> None: 
        # Initializes the convolutional layer with the given parameters

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.strip().lower()
        self.activation = activation_func(activation)

    def init_params(self, input_size: tuple) -> None:
        # Initialises the kernel & bias parameters for the convolution given the expected output size

        self.input_size = input_size
        self.output, self.padding_amount = self.calculate_size(input_size)
        self.kernel_shape = (self.kernel_size[0], self.kernel_size[1], self.input_size[3], self.filters)
        self.kernel_params = np.random.uniform(-0.5, 0.5, self.kernel_shape)
        self.bias_params = np.zeros((self.output.shape[1], self.output.shape[2], self.output.shape[3]))

    def calculate_size(self, input_size: tuple) -> tuple:
        # Calculates the expected output size and the amount of padding required on each x and y

        if self.padding == 'valid':
            output_width = np.ceil((input_size[1] - self.kernel_size[0] + 1) / self.strides).astype(int)
            output_height = np.ceil((input_size[2] - self.kernel_size[1] + 1) / self.strides).astype(int)
            return np.zeros([input_size[0], output_width, output_height, self.filters]), (0, 0)
        
        elif self.padding == 'same': 
            output_width = np.ceil(input_size[1] / self.strides).astype(int)
            output_height = np.ceil(input_size[2] / self.strides).astype(int)
            pad_width = np.ceil(((output_width - 1) * self.strides + self.kernel_size[0] - input_size[1]) / 2).astype(int)
            pad_height = np.ceil(((output_height - 1) * self.strides + self.kernel_size[1] - input_size[2]) / 2).astype(int) 
            return np.zeros((input_size[0], output_height, output_width, self.filters)), (pad_height, pad_width)

    def correlate(self, input, output, kernel_params, forward=True):
        # General function to perform a correlation between a tensor and kernel

        image_patches = np.lib.stride_tricks.sliding_window_view(
            x=input,
            window_shape=(kernel_params.shape[0], kernel_params.shape[1], kernel_params.shape[2]),   
            axis=(1, 2, 3)
        )

        if forward: 
            image_patches = image_patches[:, ::self.strides, ::self.strides, :, :, :, :]

        image_patches = image_patches.reshape(output.shape[0], output.shape[1], output.shape[2], -1)
        kernel = kernel_params.reshape(-1, kernel_params.shape[3])
        
        for i in range(output.shape[3]):
            output[:, :, :, i] = np.tensordot(image_patches, kernel[:, i], axes=([3], [0]))

        return output

    def __call__(self, input: np.ndarray) -> np.ndarray:
        # Performs the convolution and applies the activation function
        
        self.input = input
        input = np.pad(input, pad_width=((0, 0), (self.padding_amount[0], self.padding_amount[0]), (self.padding_amount[1], self.padding_amount[1]), (0, 0))) 
        self.output = self.correlate(input, self.output, self.kernel_params)
        
        return self.activation(self.output)

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Backwards function which updates the weight and bias parameters, calculates the input gradient for the next layer

        output_gradient = self.activation.backward(output_gradient)
        kernel_gradient = np.zeros_like(self.kernel_params)
        input_gradient = np.zeros_like(self.input)

        pad_x = np.ceil((self.kernel_size[0] + input_gradient.shape[1] - output_gradient.shape[1] - 1) // 2).astype(int)
        pad_y = np.ceil((self.kernel_size[1] + input_gradient.shape[2] - output_gradient.shape[2] - 1) // 2).astype(int)

        padded_output_gradient = np.pad(output_gradient, ((0, 0), (pad_x, pad_x), (pad_y, pad_y), (0, 0)))
        input_gradient = self.correlate(padded_output_gradient, input_gradient, self.kernel_params.transpose((1, 0, 3, 2)), forward=False)

        if self.strides > 1:
            output_strided = np.zeros((output_gradient.shape[0], output_gradient.shape[1] * self.strides, output_gradient.shape[2] * self.strides, output_gradient.shape[3]), dtype=np.float32)
            output_strided[:, 1::self.strides, 1::self.strides, :] = output_gradient
        else:
            output_strided = output_gradient

        for batch in range(input_gradient.shape[0]):
            for filter in range(self.filters):
                for input_filter in range(self.input_size[3]):
                    kernel_gradient[:, :, input_filter, filter] += signal.correlate2d(self.input[batch, :, :, input_filter], output_strided[batch, :, :, filter], mode='valid')
       
        kernel_gradient /= input_gradient.shape[0]            
        self.kernel_params -= learning_rate * kernel_gradient
        self.bias_params -= learning_rate * np.mean(output_gradient, axis=0) 

        return input_gradient