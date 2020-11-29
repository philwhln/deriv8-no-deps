import pytest

from deriv8.activation import relu, sigmoid


@pytest.mark.parametrize("input, expected_output", [
    ([[-1., 0., 1.], [0., 123.2, -12.56]], [[0., 0., 1.], [0., 123.2, 0.]])
])
def test_relu(input, expected_output):
    assert relu(input) == expected_output


@pytest.mark.parametrize("input, expected_output", [
    ([[-1., 0., 1.], [0., 123.2, -12.56]],
     [[0.2689414213699951, 0.5, 0.7310585786300049], [0.5, 1., 3.509617468974921e-06]])
])
def test_sigmoid(input, expected_output):
    assert sigmoid(input) == expected_output
