from deriv8.matrix2d import Matrix2D


def relu(A: Matrix2D) -> Matrix2D:
    return [[max(Aij, 0.) for Aij in Ai] for Ai in A]
