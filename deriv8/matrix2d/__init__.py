import random

Matrix2D = list[list[float]]
Shape2D = tuple[int, int]


def matrix_multiply(A: Matrix2D, B: Matrix2D) -> Matrix2D:
    A_shape = shape(A)
    B_shape = shape(B)
    assert A_shape[1] == B_shape[0], "matrices cannot be multiplied due to incompatible shapes"

    C_shape = (A_shape[0], B_shape[1])
    C = zeros(*C_shape)

    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            for k in range(B_shape[1]):
                C[i][k] += A[i][j] * B[j][k]

    return C


# naive broadcasting
def broadcast(A: Matrix2D, to_shape: Shape2D) -> Matrix2D:
    A_shape = shape(A)
    B = A

    assert (A_shape[0] == to_shape[0] and A_shape[1] == 1) or (A_shape[0] == 1 and A_shape[1] == to_shape[1]), \
        "matrix cannot be broadcast to given shape"

    if A_shape[0] == 1 and A_shape[0] < to_shape[0]:
        # expand number of rows
        B = A * to_shape[0]
    if A_shape[1] == 1 and A_shape[1] < to_shape[1]:
        # expand number of cols
        B = [A[i] * to_shape[1] for i in range(to_shape[0])]

    return B


def broadcast_A_or_B(A, B: Matrix2D) -> tuple[Matrix2D, Matrix2D]:
    A_shape = shape(A)
    B_shape = shape(B)

    if A_shape == B_shape:
        return A, B

    assert (A_shape[0] == B_shape[0] and (A_shape[1] == 1 or B_shape[1] == 1)) or \
           (A_shape[1] == B_shape[1] and (A_shape[0] == 1 or B_shape[0] == 1)), \
        "matrix shapes {} and {} are incompatible for broadcasting".format(A_shape, B_shape)

    to_shape = (max(A_shape[0], B_shape[0]), max(A_shape[1], B_shape[1]))

    if A_shape != to_shape:
        return broadcast(A, to_shape), B
    else:
        return A, broadcast(B, to_shape)


def add(A: Matrix2D, B: Matrix2D) -> Matrix2D:
    A_, B_ = broadcast_A_or_B(A, B)

    C_shape = shape(A_)
    C = zeros(*C_shape)

    for j in range(C_shape[1]):
        for i in range(C_shape[0]):
            C[i][j] = A_[i][j] + B_[i][j]

    return C


def minus(A: Matrix2D, B: Matrix2D) -> Matrix2D:
    A_, B_ = broadcast_A_or_B(A, B)

    C_shape = shape(A_)
    C = zeros(*C_shape)

    for j in range(C_shape[1]):
        for i in range(C_shape[0]):
            C[i][j] = A_[i][j] - B_[i][j]

    return C


def element_multiply(A: Matrix2D, B: Matrix2D) -> Matrix2D:
    A_, B_ = broadcast_A_or_B(A, B)

    C_shape = shape(A_)
    C = zeros(*C_shape)

    for j in range(C_shape[1]):
        for i in range(C_shape[0]):
            C[i][j] = A_[i][j] * B_[i][j]

    return C


def one_hot_encode(A: Matrix2D, labels: list) -> Matrix2D:
    A_shape = shape(A)
    assert A_shape[0] == 1

    num_labels = len(labels)
    num_values = A_shape[1]

    B = zeros(num_labels, num_values)
    for j, value in enumerate(A[0]):
        i = labels.index(value)
        B[i][j] = 1.

    return B


def rand(rows: int, cols: int) -> Matrix2D:
    return [[random.uniform(-1., 1.) for i in range(cols)] for j in range(rows)]


def shape(A: Matrix2D) -> Shape2D:
    rows = len(A)
    cols = len(A[0])  # assume all rows are equal length
    return rows, cols


def transpose(A: Matrix2D) -> Matrix2D:
    A_shape = shape(A)
    A_transposed = zeros(*reversed(A_shape))
    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            A_transposed[j][i] = A[i][j]
    return A_transposed


def zeros(rows: int, cols: int) -> Matrix2D:
    return [[0. for i in range(cols)] for j in range(rows)]
