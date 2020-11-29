from math import exp

from deriv8.matrix2d import Matrix2D, element_exp, shape, zeros


def relu(A: Matrix2D) -> Matrix2D:
    return [[max(Aij, 0.) for Aij in Ai] for Ai in A]


def sigmoid(A: Matrix2D) -> Matrix2D:
    return [[(1. / (1. + exp(-Aij))) for Aij in Ai] for Ai in A]


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
