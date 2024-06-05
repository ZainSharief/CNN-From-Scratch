import numpy as np

class binary_crossentropy:
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Calculates the binary cross-entropy loss
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))    
    
    def dervivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Calculates the dervivative of binary cross-entropy loss
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
    
class mse:
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Calculates the mean-squared error loss
        return np.mean(np.power(y_true - y_pred, 2))
    
    def dervivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Calculates the dervivative of mean-squared error loss
        return (2 * (y_pred - y_true)) / np.size(y_pred)

class categorical_crossentropy:
    # Currently not working 
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Calculates the categorical crossentropy loss
        y_pred = np.clip(y_pred, 1e-8, 1-1e-8)
        cross_entropy = -np.sum(y_true * np.log(y_pred), axis=-1)
        return np.mean(cross_entropy)

    def dervivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Calculates the dervivative of categorical crossentropy loss
        y_pred = np.clip(y_pred, 1e-8, 1-1e-8)
        gradient = y_true / y_pred
        return gradient
