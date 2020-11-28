import pytest

from deriv8.activation import relu


@pytest.mark.parametrize("input, expected_output", [
    ([[-1., 0., 1.], [0., 123.2, -12.56]], [[0., 0., 1.], [0., 123.2, 0.]])
])
def test_relu(input, expected_output):
    assert relu(input) == expected_output
