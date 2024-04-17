import numpy as np
import tensorflow as tf

class binary_crossentropy:
    def __call__(self, y_pred, y_true):
        # Calculates the binary cross-entropy loss
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))    
    
    def dervivative(self, y_pred, y_true):
        # Calculates the dervivative of binary cross-entropy loss
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
    
class mse:
    def __call__(self, y_pred, y_true):
        # Calculates the mean-squared error loss
        return np.mean(np.power(y_true - y_pred, 2))
    
    def dervivative(self, y_true, y_pred):
        # Calculates the dervivative of mean-squared error loss
        return 2 * (y_pred - y_true) / np.size(y_true)
    

            
