from typing import NoReturn

from .datasets import mnist
from .datasets.utils import shuffle_truncate_dataset
from .matrix2d import one_hot_encode, shape, transpose
from deriv8 import model


def main() -> NoReturn:
    learning_rate = 1e-1
    lamb = 1e-1
    batch_size = 500
    hidden_units = 32
    epochs = 20
    max_training_examples = 60000
    max_test_examples = 10000

    print("Loading data...")

    X_train, Y_train, X_test, Y_test = mnist.load()

    print("Using {}/{} random training examples and {}/{} random test examples"
          .format(max_training_examples, shape(X_train)[0], max_test_examples, shape(X_test)[0]))

    print("Preparing data...")

    # We want examples stacked in columns, not rows
    X_train = transpose(X_train)
    Y_train = transpose(Y_train)
    X_test = transpose(X_test)
    Y_test = transpose(Y_test)

    # shuffle and truncate
    X_train, Y_train = shuffle_truncate_dataset(X_train, Y_train, truncate=max_training_examples)
    X_test, Y_test = shuffle_truncate_dataset(X_test, Y_test, truncate=max_test_examples)

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

    model.train(X_train, Y_train, X_test, Y_test, parameters, epochs, batch_size, learning_rate, lamb)
