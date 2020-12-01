from .matrix2d import one_hot_encode, shape, transpose
from .datasets import mnist
from deriv8 import model


def main():
    learning_rate = 1e-3
    batch_size = 13
    hidden_units = 500
    epochs = 20

    print("Loading data")

    X_train, Y_train, X_test, Y_test = mnist.load()

    print("Preparing data")

    labels = list(map(float, range(10)))

    # We want examples stacked in columns, not rows
    X_train = mnist.normalize_inputs(transpose(X_train))
    Y_train = one_hot_encode(transpose(Y_train), labels)
    X_test = mnist.normalize_inputs(transpose(X_test))
    Y_test = one_hot_encode(transpose(Y_test), labels)

    input_num_units = shape(X_train)[0]
    output_num_units = shape(Y_train)[0]

    layers_num_units = [hidden_units, output_num_units]
    parameters = model.init_parameters(input_num_units, layers_num_units)

    model.train(X_train, Y_train, X_test, Y_test, parameters, epochs, batch_size, learning_rate)
