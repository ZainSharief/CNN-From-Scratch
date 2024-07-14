# Convolutional Neural Network Implementation in Python using NumPy
## Overview
This project is an implementation of a Convolutional Neural Network (CNN) from scratch using Python and NumPy. It was implemented to demonstrate my understanding of deep learning concepts and practical skills in machine learning and mathematical computing.

## Table Of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Architecture](#architecture)
- [Results](#results)
- [Examples](#examples)
- [License](#license)
- [Contact](#contact)

## Features

#### Convolutional layer Implementation
Within the project, we use two different approaches to perform the correlation operation to maximise efficiency:

- Sliding Window: During the forward pass and the calculation of the input gradient in the backwards pass, we use a sliding window which is more efficient for a small kernel size. This process leverages array slicing to apply strides and the dot-product to perform the convolution.

- Fast Fourier Transforms (FFTs): When calculating the kernel gradient, we utilise FFTs due to the large size of the two tensors being correlated since it is computationally more efficient than the direct spatial domain convolution methods.

This approach was taken, rather than typical implementations of a Convolutional Layer which use the signal library to avoid limitations with large-scale convolution operations, providing a much faster and more satisfying solution.

#### Max-Pooling Implementation
A typical max-pooling layer would consist of a pass to find the maximum item and a pass to find the position of that item, which is inefficient.

- Once again, by utilising a sliding window, we can create a view of the tensor so we only have to pass through it a single time. We can then find the maxiumum value and its respective position with ease.

Another suggestion is to use indexing abilites to find the position of the max item and use that to find the max item. However, numpy reqiures unravel_index for higher order indices which ends up being slower than seperately calculating the max and argmax. 

#### Fully Connected (Dense) Implementation


## Installation
1. Clone the repository:
   
  ```sh
  git clone https://github.com/ZainSharief/cnn-numpy.git
  ```

2. Navigate to the project directory:

  ```sh
  cd cnn-numpy
  ```

3. Install the required dependencies:

  ```sh
  pip install -r requirements.txt
  ```
## Architecture

The implemented Convolutional Neural Network architecture, provided with a Stochastic Gradient Descent optimizer, includes:

#### General Layers
- Convolutional Layers: (filters, kernel size, strides, padding type)
- Max Pooling Layers: (pool size, strides)
- Fully Connected (Dense) Layers: (units)
- Batch Normalisation: (epsilon, momentum)
- Dropout: (dropout rate)
- Flatten

#### Activation Functions
- reLU
- sigmoid
- softmax

#### Loss Functions
- Mean Squared Error
- Binary Cross-Entropy
- Categorical Cross-Entropy

## Results


## Examples
Some examples of how this script can be used are contained within these files:

[MNIST_model.py](MNIST_model.py): Example script to train on the MNIST dataset

[CIFAR100_model.py](CIFAR100_model.py): Example script to train on the CIFAR100 dataset

## License
This project is licensed under the MIT License. See the [LICENSE.txt](LICENSE.txt) file for more details.

## Contact
Zain Sharief - zain.sharief21@gmail.com - [LinkedIn](https://www.linkedin.com/in/zain-sharief-5193b425b/)

Project Link: https://github.com/ZainSharief/cnn-numpy
