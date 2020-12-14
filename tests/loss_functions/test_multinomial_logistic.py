from math import log

import pytest

from deriv8.loss_functions.multinomial_logistic import loss

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
            [[0., 0., 0.]]
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
            [[-log(0.5), -log(0.5), -log(0.5)]]
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
            [[-log(tiny), -log(tiny), -log(tiny)]]
    )
])
def test_loss(Y, Y_hat, expected_loss):
    assert loss(Y_hat, Y) == expected_loss
