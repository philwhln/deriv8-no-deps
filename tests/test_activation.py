import pytest

from deriv8.activation_functions import relu, sigmoid, softmax


@pytest.mark.parametrize("input, expected_output", [
    ([[-1., 0., 1.], [0., 123.2, -12.56]], [[0., 0., 1.], [0., 123.2, 0.]])
])
def test_relu(input, expected_output):
    assert relu.relu(input) == expected_output


@pytest.mark.parametrize("input, expected_output", [
    ([[-1., 0., 1.], [0., 123.2, -12.56]],
     [[0.2689414213699951, 0.5, 0.7310585786300049], [0.5, 1., 3.509617468974921e-06]])
])
def test_sigmoid(input, expected_output):
    assert sigmoid.sigmoid(input) == expected_output


@pytest.mark.parametrize("input, stable_softmax, expected_output", [
    ([[1., 0., 1.], [2., 123.2, -12.56]], False,
     [[0.2689414213699951, 3.1255023481538206e-54, 0.9999987088810225],
      [0.7310585786300049, 1., 1.2911189775612279e-06]]),
    ([[1., 0., 1.], [2., 123.2, -12.56]], True,
     [[0.2689414213699951, 3.1255023481538206e-54, 0.9999987088810224],
      [0.7310585786300049, 1., 1.2911189775612279e-06]]),
])
def test_softmax(input, stable_softmax, expected_output):
    assert softmax.softmax(input, stable_softmax) == expected_output
