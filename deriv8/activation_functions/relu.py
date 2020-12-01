from deriv8.matrix2d import Matrix2D


def relu(A: Matrix2D) -> Matrix2D:
    return [[max(Aij, 0.) for Aij in Ai] for Ai in A]


def relu_derivative(Z: Matrix2D):
    return [[1. if Zij > 0. else 1. for Zij in Zi] for Zi in Z]