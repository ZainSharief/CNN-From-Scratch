import tensorflow as tf
from train import *

model = [
    # Model to test the fully connected layers
    flatten(),
    dense(units=128, activation='relu'),
    dense(units=64, activation='relu'),
    dense(units=10, activation='sigmoid'),
]

# Extracts the MNIST dataset from tensorflow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocesses the dataset images
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Preprocesses the dataset annotations
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Compiles the model, creating all variables
compile(
    model=model, 
    input_shape=(32, 28, 28, 1)
)

# Training the model
train(
    model=model,
    x_train=x_train,
    y_train=y_train,
    x_val=None,
    y_val=None,
    #loss_function=,
    learning_rate=0.001,
    epochs=100
)


