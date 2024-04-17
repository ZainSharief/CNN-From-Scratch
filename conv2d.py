import numpy as np
from activation import *

class conv2d:
    def __init__(self, filters, kernel_size=(3,3), strides=1, padding='valid', activation=None):
        # Initalises all base parameters for the convolutional layer object
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.strip().lower()
        self.activation = activation

    def init_params(self, input_size):
        # Initialises the kernel parameters and biases for the convolution
        self.input_size = input_size
        total_params = self.filters * (self.kernel_size[0]*self.kernel_size[1])
        kernel_params = np.random.uniform(-0.5, 0.5, total_params)
        self.kernel_params = np.reshape(kernel_params, (self.kernel_size[0], self.kernel_size[1], self.filters))
        self.bias_params = np.random.uniform(-0.5, 0.5, self.filters)

        self.output, self.padding_amount = self.calculate_size(input_size)

    def calculate_size(self, input_size):
        # Calculates the expected output size and the amount of padding required on each x and y
        new_xsize = int((input_size[1] - self.kernel_size[0]) / self.strides) + 1
        new_ysize = int((input_size[2] - self.kernel_size[1]) / self.strides) + 1

        if self.padding == 'valid':
            return np.zeros([input_size[0], new_xsize, new_ysize, self.filters]), None
        
        elif self.padding == 'same': 
            new_xsizepool = int(np.ceil(input_size[1] / self.strides))
            new_ysizepool = int(np.ceil(input_size[2] / self.strides))

            pad_x = int(np.ceil((new_xsizepool - new_xsize) / 2))
            pad_y = int(np.ceil((new_ysizepool - new_ysize) / 2))

            return np.zeros([input_size[0], new_xsizepool, new_ysizepool, self.filters]), (pad_x, pad_y)

    def __call__(self, input):
        # Performs the convolution and applies the activation layer
        self.input = input
        if self.padding == 'same': input = np.pad(input, pad_width=((0, 0), (self.padding_amount[0], self.padding_amount[0]), (self.padding_amount[1], self.padding_amount[1]), (0, 0))) 
              
        for y in range(self.output.shape[2]):
            for x in range(self.output.shape[1]):
                x_pos = x * self.strides
                y_pos = y * self.strides

                image_kernel = input[:, x_pos:x_pos+self.kernel_size[0], y_pos:y_pos+self.kernel_size[1], :]
                self.output[:, x, y, :] = np.sum(np.tensordot(image_kernel, self.kernel_params, axes=([1,2],[0,1])), axis=1) + self.bias_params

        if self.activation == 'relu':
            return reLU()(self.output)

        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # NOT WORKING 
        kernel_gradient = np.zeros(self.kernel_params.shape)
        input_gradient = np.zeros(self.input_size[-1])

        output_gradientPADDED = np.pad(output_gradient[j], ((1, 1),(1, 1)))
        kernel_params = self.kernel_params.T

        for minibatch in range(self.input_size[0]):
            for i in range(self.filters):
                for j in range(self.input_size[-1]):

                    kernel_gradient[i, j] += np.tensordot(self.input[minibatch, :, :, j], output_gradient[i], axes=([0,1],[0,1]))
                    input_gradient[j] += np.tensordot(output_gradientPADDED[i], kernel_params[i, j], axes=([0,1],[0,1]))

        kernel_gradient /= self.input.shape[0]
        input_gradient /= self.input.shape[0]

        self.kernel_params -= learning_rate * kernel_gradient
        self.bias_params -= learning_rate * output_gradient

        return input_gradient