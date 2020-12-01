from math import exp

from deriv8 import Matrix2D
from deriv8.matrix2d import Matrix2D, element_exp, shape, zeros


def relu(A: Matrix2D) -> Matrix2D:
    return [[max(Aij, 0.) for Aij in Ai] for Ai in A]


def relu_derivative(Z: Matrix2D):
    return [[1. if Zij > 0. else 1. for Zij in Zi] for Zi in Z]


def sigmoid(A: Matrix2D) -> Matrix2D:
    return [[(1. / (1. + exp(-Aij))) for Aij in Ai] for Ai in A]


def sigmoid_derivative(Z: Matrix2D):
    return [[Zij * (1 - Zij) for Zij in Zi] for Zi in Z]


# TODO: implement stable-softmax à la https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
def softmax(A: Matrix2D) -> Matrix2D:
    A_shape = shape(A)
    A_exp = element_exp(A)
    A_exp_col_sum = zeros(1, A_shape[1])

    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            A_exp_col_sum[0][j] += A_exp[i][j]

    A_softmax = zeros(*A_shape)
    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            A_softmax[i][j] = A_exp[i][j] / A_exp_col_sum[0][j]

    return A_softmax
