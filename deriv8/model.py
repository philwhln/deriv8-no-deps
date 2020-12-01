import os
import time

from deriv8.datasets.utils import shuffle_dataset, split_into_batches
from deriv8.matrix2d import (Matrix2D, add, argmax, element_equals, element_multiply, matrix_multiply, minus, rand,
                             shape, sum_all, sum_rows, transpose, zeros)
from deriv8.loss_functions import multinomial_logistic
from deriv8.activation_functions import relu, softmax

DEBUG = bool(os.environ.get("DEBUG", False))


def init_parameters(input_num_units: int, layers_num_units: list[int]) -> dict[str, Matrix2D]:
    layers = [input_num_units, *layers_num_units]
    parameters = {}
    for layer_index in range(1, len(layers)):
        parameters["W" + str(layer_index)] = _init_layer_weights(layers[layer_index - 1], layers[layer_index])
        parameters["B" + str(layer_index)] = _init_layer_biases(layers[layer_index])
    return parameters


def _init_layer_weights(fan_in: int, fan_out: int) -> Matrix2D:
    # random numbers between -1 and 1 in a (fan_out x fan_in) size matrix
    return rand(fan_out, fan_in)


def _init_layer_biases(fan_out: int) -> Matrix2D:
    # zeros in a (fan_out x 1) size matrix
    return zeros(fan_out, 1)


def _forward_propagation(X: Matrix2D, parameters: dict[str, Matrix2D]) -> tuple[Matrix2D, dict[str, Matrix2D]]:
    W1 = parameters["W1"]
    B1 = parameters["B1"]
    W2 = parameters["W2"]
    B2 = parameters["B2"]

    A0 = X
    Z1 = add(matrix_multiply(W1, A0), B1)
    A1 = relu.relu(Z1)
    Z2 = add(matrix_multiply(W2, A1), B2)
    A2 = softmax.softmax(Z2)

    cache = {
        "A1": A1,
        "Z1": Z1,
        "A2": A2,
        "Z2": Z2,
    }

    # return predictions (activation of final layer) and cache of values along the way, which will be used for
    # backward propagation
    return A2, cache


def _calculate_cost(Y_hat, Y: Matrix2D) -> float:
    batch_size = shape(Y)[1]
    loss = multinomial_logistic.loss(Y_hat, Y)
    # average loss
    cost = (1. / batch_size) * sum_all(loss)
    return cost


def _calculate_accuracy(X, Y: Matrix2D, parameters: dict[str, Matrix2D]) -> float:
    Y_shape = shape(Y)
    Y_hat, _ = _forward_propagation(X, parameters)
    num_examples = Y_shape[1]
    num_correct = sum_all(element_equals(argmax(Y_hat), argmax(Y)))
    return num_correct / num_examples


def _backward_propagation(X, Y: Matrix2D, parameters, cache: dict[str, Matrix2D]) -> dict[str, Matrix2D]:
    X_shape = shape(X)

    batch_size = X_shape[1]

    W1 = parameters["W1"]
    B1 = parameters["B1"]
    W2 = parameters["W2"]
    B2 = parameters["B2"]

    A0 = X
    A1 = cache["A1"]
    Z1 = cache["Z1"]
    A2 = cache["A2"]
    Z2 = cache["Z2"]

    dA2 = multinomial_logistic.loss_derivative(A2, Y)
    dA2_dZ2 = softmax.softmax_derivative(A2)
    assert shape(dA2_dZ2) == (shape(A2)[1], shape(A2)[1])

    # Layer 2 derivatives
    dZ2 = matrix_multiply(dA2, dA2_dZ2)
    assert shape(dZ2) == shape(Z2)
    dW2 = element_multiply([[1. / batch_size]], matrix_multiply(dZ2, transpose(A1)))
    assert shape(dW2) == shape(W2)
    dB2 = element_multiply([[1. / batch_size]], sum_rows(dZ2))
    assert shape(dB2) == shape(B2)

    # Layer 1 derivatives
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
    }
    return gradients


def _update_parameters(parameters, gradients: dict[str, Matrix2D], learning_rate: float) -> dict[str, Matrix2D]:
    updated_parameters = {}
    for param in ("W1", "B1", "W2", "B2"):
        updated_parameters[param] = minus(parameters[param],
                                          element_multiply([[learning_rate]], gradients["d" + param]))
    return updated_parameters


def _train_one_mini_batch(X_train_batch, Y_train_batch, learning_rate, parameters):
    Y_hat, cache = _forward_propagation(X_train_batch, parameters)
    if DEBUG:
        print("Predictions : {}".format(argmax(Y_hat)))
    loss = _calculate_cost(Y_hat, Y_train_batch)
    train_accuracy = _calculate_accuracy(X_train_batch, Y_train_batch, parameters)
    gradients = _backward_propagation(X_train_batch, Y_train_batch, parameters, cache)
    parameters = _update_parameters(parameters, gradients, learning_rate)
    return loss, parameters, train_accuracy


def _train_one_epoch(X_train_batches, Y_train_batches, parameters, learning_rate):
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
    return loss, parameters


def train(X_train, Y_train, X_test, Y_test, parameters, epochs, batch_size, learning_rate):
    X_train, Y_train = shuffle_dataset(X_train, Y_train)
    X_train_batches = split_into_batches(X_train, batch_size)
    Y_train_batches = split_into_batches(Y_train, batch_size)
    print("Training")
    for epoch in range(epochs):
        print("epoch: {}".format(epoch))
        epoch_start_time = time.time()

        loss, parameters = _train_one_epoch(X_train_batches, Y_train_batches, parameters, learning_rate)

        test_accuracy = _calculate_accuracy(X_test, Y_test, parameters)

        print(" training loss: {:0.2f}  test accuracy: {:0.2f}%  duration: {:0.2f}s"
              .format(loss, test_accuracy * 100., time.time() - epoch_start_time))


