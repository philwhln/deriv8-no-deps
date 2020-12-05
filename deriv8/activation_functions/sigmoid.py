from math import exp

from deriv8.matrix2d import Tensor2D


def sigmoid(A: Tensor2D) -> Tensor2D:
    return [[(1. / (1. + exp(-Aij))) for Aij in Ai] for Ai in A]


def sigmoid_derivative(Z: Tensor2D):
    return [[Zij * (1 - Zij) for Zij in Zi] for Zi in Z]