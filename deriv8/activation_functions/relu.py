from deriv8.matrix2d import Tensor2D


def relu(A: Tensor2D) -> Tensor2D:
    return [[max(Aij, 0.) for Aij in Ai] for Ai in A]


def relu_derivative(Z: Tensor2D):
    return [[1. if Zij > 0. else 1. for Zij in Zi] for Zi in Z]