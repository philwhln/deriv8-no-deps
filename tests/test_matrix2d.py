from math import exp, sqrt

import pytest

from deriv8.matrix2d import (add, divide, element_equals, element_exp, element_log, element_multiply,
                             element_multiply_log, l2_norm, matrix_multiply, minus, negate, one_hot_encode,
                             shape, sum_all, sum_cols, sum_rows, transpose)


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
    ([[4., 10., 18.]], [[1., 2., 3.]], [[4., 5., 6.]]),
    ([[4., 5., 6.]], [[1.]], [[4., 5., 6.]]),
    ([[12.]], [[2., 3., 4.]], [[6., 4., 3.]]),
    ([[-12.], [15.], [-18.]], [[-3.]], [[4.], [-5.], [6.]]),
])
def test_element_divide(A, B, expected_C):
    assert divide(A, B) == expected_C


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


@pytest.mark.parametrize("A, expected", [
    ([[2., -0.333], [-10., 0.1]], [[-2., 0.333], [10., -0.1]]),
])
def test_negate(A, expected):
    assert negate(A) == expected


@pytest.mark.parametrize("A, B, expected_C", [
    ([[2., 3.], [10., 12.]], [[-2., 3.], [10., 11.]], [[0., 1.], [1., 0.]]),
])
def test_element_equals(A, B, expected_C):
    assert element_equals(A, B) == expected_C


@pytest.mark.parametrize("A, expected", [
    ([[2., 3.], [10., 12.]], [[exp(2.), exp(3.)], [exp(10.), exp(12.)]]),
])
def test_element_exp(A, expected):
    assert element_exp(A) == expected


@pytest.mark.parametrize("A, expected", [
    ([[exp(2.), exp(3.)], [exp(10.), exp(12.)]], [[2., 3.], [10., 12.]]),
])
def test_element_log(A, expected):
    assert element_log(A) == expected


@pytest.mark.parametrize("A, B, expected", [
    ([[1., 0.], [0., 5.]], [[exp(2.), exp(3.)], [0., exp(12.)]], [[2., 0.], [0., 60.]]),
])
def test_element_multiply_log(A, B, expected):
    assert element_multiply_log(A, B) == expected


@pytest.mark.parametrize("A, expected", [
    ([[6.], [8.], [4.], [2.], [4.], [6.]], sqrt(172.)),
])
def test_flattened_l2_norm(A, expected):
    assert l2_norm(A) == expected


@pytest.mark.parametrize("A, labels, expected", [
    ([[6., 8., 4.]], [2., 4., 6., 8.], [[0., 0., 0.], [0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]),
    ([["c", "a", "b"]], ["a", "b", "c", "d"], [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.], [0., 0., 0.]]),
])
def test_one_hot_encode(A, labels, expected):
    assert one_hot_encode(A, labels) == expected


@pytest.mark.parametrize("matrix, expected_sum", [
    ([[1., 2.], [4., 5.]], 12.),
    ([[1., 2.]], 3.),
    ([[1.], [2.]], 3.),
])
def test_sum_all(matrix, expected_sum):
    assert sum_all(matrix) == expected_sum


@pytest.mark.parametrize("matrix, expected_sums", [
    ([[1., 2.], [4., 5.]], [[3.], [9.]]),
    ([[1., 2.]], [[3.]]),
    ([[1.], [2.]], [[1.], [2.]]),
])
def test_sum_rows(matrix, expected_sums):
    assert sum_rows(matrix) == expected_sums


@pytest.mark.parametrize("matrix, expected_sums", [
    ([[1., 2.], [4., 5.]], [[5., 7.]]),
    ([[1., 2.]], [[1., 2.]]),
    ([[1.], [2.]], [[3.]]),
])
def test_sum_cols(matrix, expected_sums):
    assert sum_cols(matrix) == expected_sums
