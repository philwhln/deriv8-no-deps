from .datasets.utils import shuffle_dataset
from .matrix2d import one_hot_encode, shape, transpose
from .datasets import mnist
from deriv8 import model


def main():
    learning_rate = 1e-3
    batch_size = 100
    hidden_units = 200
    epochs = 10

    print("Loading data")

    X_train, Y_train, X_test, Y_test = mnist.load()

    print("Preparing data")

    # We want examples stacked in columns, not rows
    X_train = transpose(X_train)
    Y_train = transpose(Y_train)
    X_test = transpose(X_test)
    Y_test = transpose(Y_test)

    # shuffle and truncate
    X_train, Y_train = shuffle_dataset(X_train, Y_train, truncate=1000)
    X_test, Y_test = shuffle_dataset(X_test, Y_test, truncate=100)

    labels = list(map(float, range(10)))

    # format inputs into from -0.5 to 0.5 (instead of 0-255)
    X_train = mnist.normalize_inputs(X_train)
    X_test = mnist.normalize_inputs(X_test)

    # one-hot encode outputs for multinomial logistic regression
    Y_train = one_hot_encode(Y_train, labels)
    Y_test = one_hot_encode(Y_test, labels)

    input_num_units = shape(X_train)[0]
    output_num_units = shape(Y_train)[0]

    layers_num_units = [hidden_units, hidden_units, output_num_units]
    parameters = model.init_parameters(input_num_units, layers_num_units)

    model.train(X_train, Y_train, X_test, Y_test, parameters, epochs, batch_size, learning_rate)
