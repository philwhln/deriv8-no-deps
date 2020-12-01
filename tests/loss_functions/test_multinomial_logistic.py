from math import log

import pytest

from deriv8.loss_functions.multinomial_logistic import loss, loss_derivative

tiny = 2.2250738585072014e-308 / 2.
almost_1 = 1. - (2. * tiny)
huge = 1. / tiny


@pytest.mark.parametrize("Y, Y_hat, expected_loss", [
    # 100% correct
    (
            # one-hot input. each column is one input, so represents Y as 1st, 3rd, 2nd classes
            [[1., 0., 0.],
             [0., 0., 1.],
             [0., 1., 0.]],
            # predictions. each column is a prediction for one input
            [[1., 0., 0.],
             [0., 0., 1.],
             [0., 1., 0.]],
            # no loss, since we got perfect answers!
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]
    ),
    # 50% correct
    (
            # one-hot input
            [[1., 0., 0.],
             [0., 0., 1.],
             [0., 1., 0.]],
            # fairly good predictions
            [[0.50, 0.25, 0.25],
             [0.25, 0.25, 0.50],
             [0.25, 0.50, 0.25]],
            # 50% loss
            [[-log(0.5), 0., 0.],
             [0., 0., -log(0.5)],
             [0., -log(0.5), 0.]]
    ),
    # 100% incorrect
    (
            # one-hot input
            [[1., 0., 0.],
             [0., 0., 1.],
             [0., 1., 0.]],
            # predictions
            [[tiny, almost_1, tiny],
             [almost_1, tiny, tiny],
             [tiny, tiny, almost_1]],
            # loss due to perfectly bad predictions
            [[-log(tiny), 0., 0.],
             [0., 0., -log(tiny)],
             [0., -log(tiny), 0.]]
    )
])
def test_loss(Y, Y_hat, expected_loss):
    assert loss(Y_hat, Y) == expected_loss


@pytest.mark.parametrize("Y, Y_hat, expected_derivative", [
    # 99.9999% correct
    (
            # one-hot input
            [[1., 0., 0.],
             [0., 0., 1.],
             [0., 1., 0.]],
            # perfectly good predictions
            [[almost_1, tiny, tiny],
             [tiny, tiny, almost_1],
             [tiny, almost_1, tiny]],
            # no loss, since we got pretty perfect answers!
            [[-1., 0., 0.],
             [0., 0., -1.],
             [0., -1., 0.]]
    ),
    # 50% correct
    (
            # one-hot input
            [[1., 0., 0.],
             [0., 0., 1.],
             [0., 1., 0.]],
            # fairly good predictions
            [[0.50, 0.25, 0.25],
             [0.25, 0.25, 0.50],
             [0.25, 0.50, 0.25]],
            # some loss as we didn't rate predictions high enough
            [[-2, 0., 0.],
             [0., 0., -2],
             [0., -2, 0.]]
    ),
    # 99.9999% incorrect
    (
            # one-hot input
            [[1., 0., 0.],
             [0., 0., 1.],
             [0., 1., 0.]],
            # perfectly bad predicts
            [[tiny, almost_1, tiny],
             [almost_1, tiny, tiny],
             [tiny, tiny, almost_1]],
            # high loss due to perfectly bad predictions
            [[-huge, 0., 0.],
             [0., 0., -huge],
             [0., -huge, 0.]]
    )
])
def test_loss_derivative(Y, Y_hat, expected_derivative):
    assert loss_derivative(Y_hat, Y) == expected_derivative
