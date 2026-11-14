

import numpy as np


def sigmoid(z):
    # Sigmoid activation function: f(z) = 1 / (1 + exp(-z))
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    # Derivative of the sigmoid function: f'(z) = f(z) * (1 - f(z))
    return a * (1.0 - a)


def softmax(z):
    # Softmax activation function: f(z)_i = exp(z_i) / sum(exp(z_j)) for all j
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True) 
