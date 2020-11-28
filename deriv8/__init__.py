from deriv8.matrix2d import Matrix2D, add, element_multiply, matrix_multiply, minus, rand, shape, transpose, zeros
from deriv8.activation import relu
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


def _forward_propagation(X: Matrix2D, parameters: dict[str, Matrix2D]) -> Matrix2D:
    W1 = parameters["W1"]
    B1 = parameters["B1"]
    W2 = parameters["W2"]
    B2 = parameters["B2"]

    A0 = X
    Z1 = add(matrix_multiply(W1, A0), B1)
    A1 = relu(Z1)
    Z2 = add(matrix_multiply(W2, A1), B2)
    A2 = relu(Z2)
    Y = A2

    return Y


def calculate_loss(Y, predictions: Matrix2D) -> float:
    # TODO: implement proper loss function
    delta = minus(Y, predictions)
    return delta


def _back_propagation(X, Y, predictions: Matrix2D, parameters, cache: dict[str, Matrix2D]) -> Matrix2D:
    for (x, y, prediction) in zip(X, Y, predictions):
        pass


def main():
    Xtrain, Ytrain, Xtest, Ytest = load_mnist()

    # We want training examples stacked in columns, not rows
    Xtrain = transpose(Xtrain)
    Ytrain = transpose(Ytrain)
    Xtest = transpose(Xtest)
    Ytest = transpose(Ytest)

    input_num_units = shape(Xtrain)[0]
    output_num_units = shape(Ytrain)[0]

    print("input_num_units={} output_num_units={}".format(input_num_units, output_num_units))

    layers_num_units = [100, output_num_units]
    parameters = _init_parameters(input_num_units, layers_num_units)

    predictions = _forward_propagation(Xtrain, parameters)

    loss = calculate_loss(Ytrain, predictions)
    print("training loss: {}".format(loss))

    print(predictions)
