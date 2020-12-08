import os
import time
from copy import deepcopy
from typing import Dict, List, Tuple, NoReturn

from deriv8.datasets.utils import shuffle_truncate_dataset, split_into_batches
from deriv8.matrix2d import (Tensor2D, add, argmax, divide, element_equals, element_multiply, l2_norm,
                             matrix_multiply, minus, rand, shape, sum_all, sum_rows, transpose, zeros)
from deriv8.loss_functions import multinomial_logistic
from deriv8.activation_functions import relu, softmax

Parameters = Dict[str, Tensor2D]

DEBUG = os.environ.get("DEBUG", "0") == "1"


def init_parameters(input_num_units: int, layers_num_units: List[int]) -> Parameters:
    layers = [input_num_units, *layers_num_units]
    parameters = {}
    for layer_index in range(1, len(layers)):
        parameters["W" + str(layer_index)] = _init_layer_weights(layers[layer_index - 1], layers[layer_index])
        parameters["B" + str(layer_index)] = _init_layer_biases(layers[layer_index])
    return parameters


def _init_layer_weights(fan_in: int, fan_out: int) -> Tensor2D:
    # random numbers between -1 and 1 in a (fan_out x fan_in) size matrix
    return rand(fan_out, fan_in)


def _init_layer_biases(fan_out: int) -> Tensor2D:
    # zeros in a (fan_out x 1) size matrix
    return zeros(fan_out, 1)


def _forward_propagation(X: Tensor2D, parameters: Parameters) -> Tuple[Tensor2D, Parameters]:
    W1 = parameters["W1"]
    B1 = parameters["B1"]
    W2 = parameters["W2"]
    B2 = parameters["B2"]
    W3 = parameters["W3"]
    B3 = parameters["B3"]

    A0 = X
    Z1 = add(matrix_multiply(W1, A0), B1)
    A1 = relu.relu(Z1)
    Z2 = add(matrix_multiply(W2, A1), B2)
    A2 = relu.relu(Z2)
    Z3 = add(matrix_multiply(W3, A2), B3)
    A3 = softmax.softmax(Z3)

    cache = {
        "A1": A1,
        "Z1": Z1,
        "A2": A2,
        "Z2": Z2,
        "A3": A3,
        "Z3": Z3,
    }

    # return predictions (activation of final layer) and cache of values along the way, which will be used for
    # backward propagation
    return A3, cache


def _calculate_cost(Y_hat, Y: Tensor2D) -> float:
    loss = multinomial_logistic.loss(Y_hat, Y)
    # average loss
    batch_size = shape(Y)[1]
    cost = (1. / batch_size) * sum_all(loss)
    return cost


def _calculate_accuracy(X, Y: Tensor2D, parameters: Parameters) -> float:
    Y_shape = shape(Y)
    Y_hat, _ = _forward_propagation(X, parameters)
    num_examples = Y_shape[1]
    num_correct = sum_all(element_equals(argmax(Y_hat), argmax(Y)))
    return num_correct / num_examples


def _single_param_numerical_gradient(X, Y: Tensor2D, parameters: Parameters, param_name: str, i, j: int,
                                     epsilon: float) -> float:
    orig_param_value = parameters[param_name][i][j]

    parameters[param_name][i][j] = orig_param_value - epsilon
    Y_hat, cache = _forward_propagation(X, parameters)
    minus_epsilon_cost = _calculate_cost(Y_hat, Y)

    parameters[param_name][i][j] = orig_param_value + epsilon
    Y_hat, cache = _forward_propagation(X, parameters)
    plus_epsilon_cost = _calculate_cost(Y_hat, Y)

    parameters[param_name][i][j] = orig_param_value

    return (plus_epsilon_cost - minus_epsilon_cost) / (2 * epsilon)


def _params_to_single_vector(parameters: Parameters) -> Tensor2D:
    size = 0
    for param_values in parameters.values():
        param_shape = shape(param_values)
        size += param_shape[0] * param_shape[1]

    vector = zeros(size, 1)

    offset = 0
    for param_name in sorted(parameters.keys()):
        param_values = parameters[param_name]
        param_shape = shape(param_values)
        for i in range(param_shape[0]):
            for j in range(param_shape[1]):
                index = offset + (j * param_shape[0]) + i
                vector[index][0] = param_values[i][j]
        offset += param_shape[0] * param_shape[1]

    return vector


def _check_gradients(X, Y: Tensor2D, parameters: Parameters, gradients: Parameters):
    epsilon = 1e-7
    parameters_ = deepcopy(parameters)
    numerical_gradients = {}
    for param_name, param_values in parameters_.items():
        print("Calculating numeric gradients for {}".format(param_name))
        param_shape = shape(param_values)
        numerical_gradients[param_name] = zeros(*param_shape)
        for i in range(param_shape[0]):
            for j in range(param_shape[1]):
                numerical_gradients[param_name][i][j] = _single_param_numerical_gradient(X, Y, parameters_, param_name,
                                                                                         i, j, epsilon)

    gradients_vector = _params_to_single_vector(gradients)
    numerical_gradients_vector = _params_to_single_vector(numerical_gradients)

    assert shape(gradients_vector) == shape(numerical_gradients_vector)

    delta = l2_norm(minus(numerical_gradients_vector, gradients_vector)) / (
            l2_norm(numerical_gradients_vector) + l2_norm(gradients_vector))

    if delta > epsilon:
        print("Gradient check failed delta={} > {} !!!!!".format(delta, epsilon))
    else:
        print("Gradient check passed delta={}".format(delta))


def _backward_propagation(X, Y: Tensor2D, parameters, cache: Parameters) -> Parameters:
    X_shape = shape(X)

    batch_size = X_shape[1]

    W1 = parameters["W1"]
    B1 = parameters["B1"]
    W2 = parameters["W2"]
    B2 = parameters["B2"]
    W3 = parameters["W3"]
    B3 = parameters["B3"]

    A0 = X
    A1 = cache["A1"]
    Z1 = cache["Z1"]
    A2 = cache["A2"]
    Z2 = cache["Z2"]
    A3 = cache["A3"]
    Z3 = cache["Z3"]
    Y_hat = A3

    # Layer 3 (output) derivatives
    dZ3 = minus(Y_hat, Y)
    assert shape(dZ3) == shape(Z3)
    dW3 = element_multiply([[1. / batch_size]], matrix_multiply(dZ3, transpose(A2)))
    assert shape(dW3) == shape(W3)
    dB3 = element_multiply([[1. / batch_size]], sum_rows(dZ3))
    assert shape(dB3) == shape(B3)

    # Layer 2 (hidden) derivatives
    dZ2 = element_multiply(matrix_multiply(transpose(W3), dZ3), relu.relu_derivative(Z2))
    assert shape(dZ2) == shape(Z2)
    dW2 = element_multiply([[1. / batch_size]], matrix_multiply(dZ2, transpose(A1)))
    assert shape(dW2) == shape(W2)
    dB2 = element_multiply([[1. / batch_size]], sum_rows(dZ2))
    assert shape(dB2) == shape(B2)

    # Layer 1 (hidden) derivatives
    dZ1 = element_multiply(matrix_multiply(transpose(W2), dZ2), relu.relu_derivative(Z1))
    assert shape(dZ1) == shape(Z1)
    dW1 = element_multiply([[1. / batch_size]], matrix_multiply(dZ1, transpose(A0)))
    assert shape(dW1) == shape(W1)
    dB1 = element_multiply([[1. / batch_size]], sum_rows(dZ1))
    assert shape(dB1) == shape(B1)

    # return gradients for weights and bias for each layer
    gradients = {
        "dW1": dW1,
        "dB1": dB1,
        "dW2": dW2,
        "dB2": dB2,
        "dW3": dW3,
        "dB3": dB3,
    }

    return gradients


def _update_parameters(parameters, gradients: Parameters, learning_rate: float) -> Parameters:
    updated_parameters = {}
    for param in ("W1", "B1", "W2", "B2", "W3", "B3"):
        updated_parameters[param] = minus(parameters[param],
                                          element_multiply([[learning_rate]], gradients["d" + param]))
    return updated_parameters


def _train_one_mini_batch(X_train_batch, Y_train_batch: Tensor2D, learning_rate: float, parameters: Parameters) \
        -> Tuple[float, Parameters, float]:
    Y_hat, cache = _forward_propagation(X_train_batch, parameters)
    loss = _calculate_cost(Y_hat, Y_train_batch)
    train_accuracy = _calculate_accuracy(X_train_batch, Y_train_batch, parameters)
    gradients = _backward_propagation(X_train_batch, Y_train_batch, parameters, cache)
    if DEBUG:
        _check_gradients(X_train_batch, Y_train_batch, parameters, gradients)
    parameters = _update_parameters(parameters, gradients, learning_rate)
    return loss, parameters, train_accuracy


def _train_one_epoch(X_train_batches, Y_train_batches: Tensor2D, parameters: Parameters,
                     learning_rate: float) -> Parameters:
    total_batches = len(X_train_batches)
    trained_examples = 0
    for batch_index in range(len(X_train_batches)):
        batch_start_time = time.time()

        X_train_batch = X_train_batches[batch_index]
        Y_train_batch = Y_train_batches[batch_index]

        loss, parameters, train_accuracy = _train_one_mini_batch(X_train_batch, Y_train_batch, learning_rate,
                                                                 parameters)

        batch_duration = time.time() - batch_start_time

        trained_examples += shape(X_train_batch)[1]

        print(" batch: {}/{}  training loss: {:0.2f}  train accuracy: {:0.2f}%  duration: {:0.2f}s"
              .format(batch_index + 1, total_batches, loss, train_accuracy * 100., batch_duration))

    return parameters


def train(X_train, Y_train, X_test, Y_test: Tensor2D, parameters: Parameters, epochs, batch_size: int,
          learning_rate: float) -> NoReturn:
    X_train, Y_train = shuffle_truncate_dataset(X_train, Y_train)
    X_train_batches = split_into_batches(X_train, batch_size)
    Y_train_batches = split_into_batches(Y_train, batch_size)
    print("Training {} mini-batches of size {} for {} epochs".format(len(X_train_batches), batch_size, epochs))
    for epoch in range(epochs):
        print("epoch: {}".format(epoch))
        epoch_start_time = time.time()

        parameters = _train_one_epoch(X_train_batches, Y_train_batches, parameters, learning_rate)

        test_start_time = time.time()
        test_accuracy = _calculate_accuracy(X_test, Y_test, parameters)

        print(" test accuracy: {:0.2f}%  duration: {:0.2f}s (test: {:0.2f}s)"
              .format(test_accuracy * 100., time.time() - epoch_start_time, time.time() - test_start_time))
