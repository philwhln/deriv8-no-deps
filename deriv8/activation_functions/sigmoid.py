from math import exp

from deriv8.matrix2d import Matrix2D


def sigmoid(A: Matrix2D) -> Matrix2D:
    return [[(1. / (1. + exp(-Aij))) for Aij in Ai] for Ai in A]


def sigmoid_derivative(Z: Matrix2D):
    return [[Zij * (1 - Zij) for Zij in Zi] for Zi in Z]