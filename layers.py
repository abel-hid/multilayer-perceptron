

import numpy as np
from activations import sigmoid, sigmoid_derivative, softmax


def he_uniform(n_inputs, n_outputs, rng):
    limit = np.sqrt(6.0 / n_inputs)
    return rng.uniform(-limit, limit, size=(n_inputs, n_outputs))


class DenseLayer:
    def __init__(self, n_inputs, n_outputs, activation="sigmoid", rng=None):
        rng = rng or np.random.default_rng()

        self.activation_name = activation
        self.weights = he_uniform(n_inputs, n_outputs, rng)
        self.bias = np.zeros((1, n_outputs))

        self.input_batch = None
        self.output_batch = None

        self.grad_weights = None
        self.grad_bias = None

        # Adam memory: m = average of the gradient, v = average of the squared gradient
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)

    def forward(self, input_batch):
        self.input_batch = input_batch
        weighted_sum = input_batch @ self.weights + self.bias

        if self.activation_name == "sigmoid":
            self.output_batch = sigmoid(weighted_sum)
        elif self.activation_name == "softmax":
            self.output_batch = softmax(weighted_sum)
        else:
            raise ValueError(f"unknown activation '{self.activation_name}'")

        return self.output_batch

    def backward(self, grad_output):

        batch_size = self.input_batch.shape[0]

        if self.activation_name == "softmax":
            grad_weighted_sum = grad_output
        elif self.activation_name == "sigmoid":
            grad_weighted_sum = grad_output * sigmoid_derivative(self.output_batch)
        else:
            raise ValueError(f"unknown activation '{self.activation_name}'")

        self.grad_weights = self.input_batch.T @ grad_weighted_sum / batch_size
        self.grad_bias = np.sum(grad_weighted_sum, axis=0, keepdims=True) / batch_size

        grad_input = grad_weighted_sum @ self.weights.T
        return grad_input


    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias


    def update_adam(self, learning_rate, step, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # weights
        self.m_weights = beta1 * self.m_weights + (1 - beta1) * self.grad_weights
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * self.grad_weights ** 2
        m_hat = self.m_weights / (1 - beta1 ** step)
        v_hat = self.v_weights / (1 - beta2 ** step)
        self.weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
 
        # bias
        self.m_bias = beta1 * self.m_bias + (1 - beta1) * self.grad_bias
        self.v_bias = beta2 * self.v_bias + (1 - beta2) * self.grad_bias ** 2
        m_hat = self.m_bias / (1 - beta1 ** step)
        v_hat = self.v_bias / (1 - beta2 ** step)
        self.bias -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)