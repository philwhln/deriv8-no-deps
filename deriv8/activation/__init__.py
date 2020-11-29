from math import exp


from deriv8.matrix2d import Matrix2D


def relu(A: Matrix2D) -> Matrix2D:
    return [[max(Aij, 0.) for Aij in Ai] for Ai in A]


def sigmoid(A: Matrix2D) -> Matrix2D:
    return [[(1. / (1. + exp(-Aij))) for Aij in Ai] for Ai in A]
