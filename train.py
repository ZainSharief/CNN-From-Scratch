import numpy as np

# Imports all written layers
from conv2d import conv2d
from maxpool2d import maxpool2d
from dense import dense
from flatten import flatten

# Imports the activation and loss functions
from activation import *
from loss import *

def compile(model, input_shape):
    # Initialises the layer weights & biases and calculates the output sizes given the input shape
    for layer in model:

        if isinstance(layer, conv2d) or isinstance(layer, dense):
            layer.init_params(input_shape)

        elif isinstance(layer, maxpool2d):
            layer.calculate_size(input_shape)

        input_shape = layer(np.zeros(input_shape)).shape

def train(model, x_train, y_train, x_val, y_val, loss_function, learning_rate=0.01, learning_rate_scheduler=1, epochs=10, batch_size=32, shuffle=True):
    
    for e in range(epochs):
        loss = 0
        correct = 0
        total = 0

        learning_rate *= learning_rate_scheduler

        if shuffle:
            # Shuffles the training data in the same way
            permutation = np.random.permutation(len(x_train))  
            x_train = x_train[permutation]
            y_train = y_train[permutation]

        for batch in range(0, len(x_train)-(len(x_train)%batch_size), batch_size):
            # Splitting the input data into batches
            x_batch = x_train[batch:batch+batch_size]
            y_batch = y_train[batch:batch+batch_size]
            
            # Forward pass 
            for layer in model:
                x_batch = layer(x_batch)

            # Calculates the loss
            loss += loss_function(x_batch, y_batch)

            # Backward pass
            grad = loss_function.dervivative(x_batch, y_batch)
            for layer in reversed(model):
                grad = layer.backward(grad, learning_rate)

            batch_num = ((batch+1)//32)+1
            print(f"epoch={e + 1}/{epochs}: batch={batch_num}/{len(x_train)//batch_size}, loss={loss/batch_num}", end="\r")
        
        for batch in range(0, len(x_val)-(len(x_val)%batch_size), batch_size):
            # Splitting the input data into batches
            x_batch = x_val[batch:batch+batch_size]
            y_batch = y_val[batch:batch+batch_size]
            
            # Forward pass 
            for layer in model:
                x_batch = layer(x_batch)

            for batch_idx in range(x_batch.shape[0]):
                batch_output = x_batch[batch_idx]
                batch_true = y_batch[batch_idx]

                if np.argmax(batch_output) == np.argmax(batch_true):
                    correct += 1
                total += 1

        print(f"\nepoch={e + 1}/{epochs}: accuracy={correct/total}")
