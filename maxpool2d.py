import numpy as np

class maxpool2d:
    def __init__(self, pool_size=(3,3), strides=2, padding='valid'):
        # Initalises all base parameters for the max pooling layer object
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding.strip().lower()

    def calculate_size(self, input_size):
        # Calculates the expected output size and the amount of padding required on each x and y
        new_xsize = int(np.ceil((input_size[1] - self.pool_size[0] + 1) / self.strides))
        new_ysize = int(np.ceil((input_size[2] - self.pool_size[1] + 1) / self.strides))

        if self.padding == 'valid':
            self.output = np.zeros([input_size[0], new_xsize, new_ysize, input_size[3]])
            self.padding_amount = None
    
        elif self.padding == 'same': 

            new_xsizepool = int(np.ceil(input_size[1] / self.strides))
            new_ysizepool = int(np.ceil(input_size[2] / self.strides))

            pad_x = int(np.ceil((new_xsizepool - new_xsize) / 2))
            pad_y = int(np.ceil((new_ysizepool - new_ysize) / 2))

            self.output = np.zeros([input_size[0], new_xsizepool, new_ysizepool, input_size[3]]) 
            self.padding_amount = (pad_x, pad_y)
    
    def __call__(self, input):
        # Performs the max pooling layer
        if self.padding == 'same': input = np.pad(input, pad_width=((0, 0), (self.padding_amount[0], self.padding_amount[0]), (self.padding_amount[1], self.padding_amount[1]), (0, 0)))     

        for y in range(self.output.shape[2]):
            for x in range(self.output.shape[1]):
                x_pos = x * self.strides
                y_pos = y * self.strides

                image_kernel = input[:, x_pos:x_pos+self.pool_size[0], y_pos:y_pos+self.pool_size[1], :]
                self.output[:, x, y, :] = np.max(image_kernel, axis=(1,2))

        return self.output