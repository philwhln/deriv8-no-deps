
from deriv8.matrix2d import Matrix2D, add, matrix_multiply, rand, zeros


def _init_parameters(input_num_units: int, layers_num_units: int) -> dict[str, Matrix2D]:
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


def _forward_propagation(X: Matrix2D, parameters: dict[str, Matrix2D]) -> Matrix2D:
    W1 = parameters["W1"]
    B1 = parameters["B1"]
    A1 = add(matrix_multiply(W1, X), B1)

    Y = A1 # temp
    return Y


def main():
    input_num_units = 2
    layers_num_units = [3, 1]
    parameters = _init_parameters(input_num_units, layers_num_units)
    print(parameters)


if __name__ == '__main__':
    main()
