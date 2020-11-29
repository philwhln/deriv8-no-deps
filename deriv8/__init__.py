import time

from deriv8.matrix2d import (Matrix2D, add, argmax, element_equals, element_log, element_multiply, element_multiply_log,
                             matrix_multiply, minus, one_hot_encode, rand, shape, sum_all, sum_rows, transpose, zeros)
from deriv8.activation import relu, softmax
from deriv8.datasets import load_mnist


def _init_parameters(input_num_units: int, layers_num_units: list[int]) -> dict[str, Matrix2D]:
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


def _relu_activation(A: Matrix2D) -> Matrix2D:
    return [[max(Aij, 0.) for Aij in Ai] for Ai in A]


def _forward_propagation(X: Matrix2D, parameters: dict[str, Matrix2D]) -> tuple[Matrix2D, dict[str, Matrix2D]]:
    W1 = parameters["W1"]
    B1 = parameters["B1"]
    W2 = parameters["W2"]
    B2 = parameters["B2"]

    A0 = X
    Z1 = add(matrix_multiply(W1, A0), B1)
    A1 = relu(Z1)
    Z2 = add(matrix_multiply(W2, A1), B2)
    A2 = softmax(Z2)

    cache = {
        "A1": A1,
        "Z1": Z1,
        "A2": A2,
        "Z2": Z2,
    }

    # return predictions (activation of final layer) and cache of values along the way, which will be used for
    # backward propagation
    return A2, cache


def calculate_cost(Y, Y_hat: Matrix2D) -> float:
    batch_size = shape(Y)[1]
    # Note: element_multiply_log skips evaluating log(y_hat) when y is zero.
    log_probs = add(element_multiply_log(Y, Y_hat),
                    element_multiply_log(minus([[1.]], Y), minus([[1.]], Y_hat)))
    cost = (-1. / batch_size) * sum_all(log_probs)
    return cost


def calculate_accuracy(X, Y: Matrix2D, parameters: dict[str, Matrix2D]) -> float:
    Y_shape = shape(Y)
    Y_hat, _ = _forward_propagation(X, parameters)
    correct = sum_all(element_equals(argmax(Y_hat), argmax(Y)))
    return (Y_shape[1] / correct)


def _relu_derivative(Z: Matrix2D):
    return [[1. if Zij > 0. else 1. for Zij in Zi] for Zi in Z]


def _backward_propagation(X, Y: Matrix2D, parameters, cache: dict[str, Matrix2D]) -> dict[str, Matrix2D]:
    batch_size = shape(X)[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A0 = X
    A1 = cache["A1"]
    Z1 = cache["Z1"]
    A2 = cache["A2"]
    Z2 = cache["Z2"]

    dZ2 = minus(A2, Y)
    assert shape(dZ2) == (10, batch_size)
    dW2 = element_multiply([[1. / batch_size]], matrix_multiply(dZ2, transpose(A1)))
    assert shape(dW2) == (10, 100)
    dB2 = element_multiply([[1. / batch_size]], sum_rows(dZ2))
    assert shape(dB2) == (10, 1)
    dZ1 = element_multiply(matrix_multiply(transpose(W2), dZ2), _relu_derivative(Z1))
    assert shape(dZ1) == (100, batch_size)
    dW1 = element_multiply([[1. / batch_size]], matrix_multiply(dZ1, transpose(A0)))
    assert shape(dW1) == (100, 784)
    dB1 = element_multiply([[1. / batch_size]], sum_rows(dZ1))
    assert shape(dB1) == (100, 1)

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


def _normalize_inputs(X: Matrix2D) -> Matrix2D:
    return minus(element_multiply(X, [[1. / 255.]]), [[0.5]])


def main():
    print("Loading data")

    Xtrain, Ytrain, Xtest, Ytest = load_mnist()

    print("Preparing data")

    labels = list(map(float, range(10)))

    # We want training examples stacked in columns, not rows
    Xtrain = _normalize_inputs(transpose(Xtrain))
    Ytrain = one_hot_encode(transpose(Ytrain), labels)
    Xtest = _normalize_inputs(transpose(Xtest))
    Ytest = one_hot_encode(transpose(Ytest), labels)

    input_num_units = shape(Xtrain)[0]
    output_num_units = shape(Ytrain)[0]

    layers_num_units = [100, output_num_units]
    parameters = _init_parameters(input_num_units, layers_num_units)

    print("Training")

    learning_rate = 1e-3

    for epoch in range(20):
        epoch_start_time = time.time()

        Y_hat, cache = _forward_propagation(Xtrain, parameters)

        loss = calculate_cost(Ytrain, Y_hat)

        train_accuracy = calculate_accuracy(Xtrain, Ytrain, parameters)
        test_accuracy = calculate_accuracy(Xtest, Ytest, parameters)

        gradients = _backward_propagation(Xtrain, Ytrain, parameters, cache)

        parameters = _update_parameters(parameters, gradients, learning_rate)

        epoch_duration = time.time() - epoch_start_time

        print("epoch: {}  training loss: {:0.2f}  train accuracy: {:0.2f}%  test accuracy: {:0.2f}%  duration: {:0.2f}s"
              .format(epoch, loss, train_accuracy, test_accuracy, epoch_duration))
