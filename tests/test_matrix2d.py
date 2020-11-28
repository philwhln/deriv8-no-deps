import pytest

from deriv8.matrix2d import add, element_multiply, matrix_multiply, minus, one_hot_encode, shape, transpose


@pytest.mark.parametrize("matrix, expected_shape", [
    ([[1., 2., 3.], [4., 5., 6.]], (2, 3)),
    ([[1., 2.], [3., 4.], [5., 6.]], (3, 2)),
])
def test_shape(matrix, expected_shape):
    assert shape(matrix) == expected_shape


@pytest.mark.parametrize("matrix, expected_transpose", [
    ([[1., 2.], [4., 5.]], [[1., 4.], [2., 5.]]),
    ([[1., 2., 3.], [4., 5., 6.]], [[1., 4.], [2., 5.], [3., 6.]]),
    ([[1., 4.], [2., 5.], [3., 6.]], [[1., 2., 3.], [4., 5., 6.]]),
])
def test_transpose(matrix, expected_transpose):
    assert transpose(matrix) == expected_transpose


@pytest.mark.parametrize("A, B, expected_C", [
    ([[6.]], [[7.]], [[42.]]),
    ([[1., 22., 333.]], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [[1., 22., 333.]]),
    ([[1., 2., 3.]], [[4.], [5.], [6.]], [[32.]]),
    ([[1., 2., 3.], [10., 20., 30.]], [[4., 50.], [15., 61.], [-5., -7.]], [[19., 151.], [190., 1510.]]),
])
def test_matrix_multiply(A, B, expected_C):
    assert matrix_multiply(A, B) == expected_C


@pytest.mark.parametrize("A, B, expected_C", [
    ([[1., 2., 3.]], [[4., 5., 6.]], [[4., 10., 18.]]),
    ([[1.]], [[4., 5., 6.]], [[4., 5., 6.]]),
    ([[4., 5., 6.]], [[2.]], [[8., 10., 12.]]),
    ([[-3.]], [[4.], [-5.], [6.]], [[-12.], [15.], [-18.]]),
    ([[4.], [5.], [6.]], [[4.]], [[16.], [20.], [24.]]),
])
def test_element_multiply(A, B, expected_C):
    assert element_multiply(A, B) == expected_C


@pytest.mark.parametrize("A, B, expected_C", [
    ([[1., 2., 3.]], [[4., 5., 6.]], [[5., 7., 9.]]),
    ([[1.]], [[4., 5., 6.]], [[5., 6., 7.]]),
    ([[4., 5., 6.]], [[1.]], [[5., 6., 7.]]),
    ([[1.]], [[4.], [5.], [6.]], [[5.], [6.], [7.]]),
    ([[4.], [5.], [6.]], [[1.]], [[5.], [6.], [7.]]),
])
def test_add(A, B, expected_C):
    assert add(A, B) == expected_C


@pytest.mark.parametrize("A, B, expected_C", [
    ([[1., 8., 7.]], [[4., 5., 6.]], [[-3., 3., 1.]]),
    ([[1.]], [[4., 5., 6.]], [[-3., -4., -5.]]),
    ([[4., 5., 6.]], [[1.]], [[3., 4., 5.]]),
    ([[1.]], [[4.], [15.], [26.]], [[-3.], [-14.], [-25.]]),
    ([[4.], [15.], [26.]], [[1.]], [[3.], [14.], [25.]]),
])
def test_minus(A, B, expected_C):
    assert minus(A, B) == expected_C


@pytest.mark.parametrize("A, labels, expected", [
    ([[6., 8., 4.]], [2., 4., 6., 8.], [[0., 0., 0.], [0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]),
    ([["c", "a", "b"]], ["a", "b", "c", "d"], [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.], [0., 0., 0.]]),
])
def test_one_hot_encode(A, labels, expected):
    assert one_hot_encode(A, labels) == expected
