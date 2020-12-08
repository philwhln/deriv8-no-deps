import random
from math import exp, log, sqrt
from typing import List, Tuple

Tensor2D = List[List[float]]
Shape2D = Tuple[int, int]


def matrix_multiply(A: Tensor2D, B: Tensor2D) -> Tensor2D:
    A_shape = shape(A)
    B_shape = shape(B)
    assert A_shape[1] == B_shape[0], \
        "matrix shapes {} and {} are incompatible for matrix multiplying".format(A_shape, B_shape)

    C_shape = (A_shape[0], B_shape[1])
    C = zeros(*C_shape)

    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            for k in range(B_shape[1]):
                C[i][k] += A[i][j] * B[j][k]

    return C


# naive broadcasting
def broadcast(A: Tensor2D, to_shape: Shape2D) -> Tensor2D:
    A_shape = shape(A)

    assert (A_shape == (1, 1)) or \
           (A_shape[0] == to_shape[0] and A_shape[1] == 1) or \
           (A_shape[0] == 1 and A_shape[1] == to_shape[1]), \
        "matrix shape {} cannot be broadcast to shape {}".format(A_shape, to_shape)

    if A_shape[0] == 1 and A_shape[0] < to_shape[0] and A_shape[1] == 1 and A_shape[1] < to_shape[1]:
        # expand number of rows and number of cols
        return [[A[0][0]] * to_shape[1]] * to_shape[0]
    elif A_shape[0] == 1 and A_shape[0] < to_shape[0]:
        # expand number of rows
        return A * to_shape[0]
    elif A_shape[1] == 1 and A_shape[1] < to_shape[1]:
        # expand number of cols
        return [A[i] * to_shape[1] for i in range(to_shape[0])]

    raise Exception("Unexpected for A_shape:{} to_shape:{}".format(A_shape, to_shape))


def broadcast_A_or_B(A, B: Tensor2D) -> Tuple[Tensor2D, Tensor2D]:
    A_shape = shape(A)
    B_shape = shape(B)

    if A_shape == B_shape:
        return A, B

    assert (A_shape == (1, 1) or B_shape == (1, 1)) or \
           (A_shape[0] == B_shape[0] and (A_shape[1] == 1 or B_shape[1] == 1)) or \
           (A_shape[1] == B_shape[1] and (A_shape[0] == 1 or B_shape[0] == 1)), \
        "matrix shapes {} and {} are incompatible for broadcasting".format(A_shape, B_shape)

    to_shape = (max(A_shape[0], B_shape[0]), max(A_shape[1], B_shape[1]))

    if A_shape != to_shape:
        return broadcast(A, to_shape), B
    else:
        return A, broadcast(B, to_shape)


def add(A: Tensor2D, B: Tensor2D) -> Tensor2D:
    A_, B_ = broadcast_A_or_B(A, B)

    C_shape = shape(A_)
    C = zeros(*C_shape)

    for j in range(C_shape[1]):
        for i in range(C_shape[0]):
            C[i][j] = A_[i][j] + B_[i][j]

    return C


def minus(A: Tensor2D, B: Tensor2D) -> Tensor2D:
    A_, B_ = broadcast_A_or_B(A, B)

    C_shape = shape(A_)
    C = zeros(*C_shape)

    for j in range(C_shape[1]):
        for i in range(C_shape[0]):
            C[i][j] = A_[i][j] - B_[i][j]

    return C


def divide(A: Tensor2D, B: Tensor2D) -> Tensor2D:
    A_, B_ = broadcast_A_or_B(A, B)

    C_shape = shape(A_)
    C = zeros(*C_shape)

    for j in range(C_shape[1]):
        for i in range(C_shape[0]):
            C[i][j] = A_[i][j] / B_[i][j]

    return C


def negate(A: Tensor2D) -> Tensor2D:
    return [[-Aij for Aij in Ai] for Ai in A]


def argmax(A: Tensor2D) -> Tensor2D:
    A_shape = shape(A)
    A_max = zeros(1, A_shape[1])
    A_argmax = zeros(1, A_shape[1])

    for j in range(A_shape[1]):
        for i in range(A_shape[0]):
            if A[i][j] > A_max[0][j]:
                A_max[0][j] = A[i][j]
                A_argmax[0][j] = i

    return A_argmax


def element_multiply(A: Tensor2D, B: Tensor2D) -> Tensor2D:
    A_, B_ = broadcast_A_or_B(A, B)

    C_shape = shape(A_)
    C = zeros(*C_shape)

    for j in range(C_shape[1]):
        for i in range(C_shape[0]):
            C[i][j] = A_[i][j] * B_[i][j]

    return C


# Even though mathematically a.log(b) will be zero when a and b are both zero, we'll still be evaluating
# log(b) and log(0) causes "ValueError: math domain error". This function combines both, to skip the log(b)
# evaluation when a is zero.
def element_multiply_log(A: Tensor2D, B: Tensor2D) -> Tensor2D:
    A_, B_ = broadcast_A_or_B(A, B)

    C_shape = shape(A_)
    C = zeros(*C_shape)

    for j in range(C_shape[1]):
        for i in range(C_shape[0]):
            if A_[i][j] == 0.:
                C[i][j] = 0.
            else:
                try:
                    C[i][j] = A_[i][j] * log(B_[i][j])
                except ValueError:
                    print("B_[i][j]={}".format(B_[i][j]))
                    raise

    return C


def element_equals(A: Tensor2D, B: Tensor2D) -> Tensor2D:
    A_, B_ = broadcast_A_or_B(A, B)

    C_shape = shape(A_)
    C = zeros(*C_shape)

    for j in range(C_shape[1]):
        for i in range(C_shape[0]):
            C[i][j] = A_[i][j] == B_[i][j]

    return C


def element_exp(A: Tensor2D) -> Tensor2D:
    return [[exp(Aij) for Aij in Ai] for Ai in A]


def element_log(A: Tensor2D) -> Tensor2D:
    return [[log(Aij) for Aij in Ai] for Ai in A]


def l2_norm(A: Tensor2D) -> float:
    A_shape = shape(A)

    # assumes a single column vector
    assert A_shape[1] == 1

    sq_sums = 0.
    for i in range(A_shape[0]):
        sq_sums += A[i][0] ** 2.

    return sqrt(sq_sums)


def one_hot_encode(A: Tensor2D, labels: list) -> Tensor2D:
    A_shape = shape(A)
    assert A_shape[0] == 1

    num_classes = len(labels)
    num_values = A_shape[1]

    B = zeros(num_classes, num_values)
    for j, value in enumerate(A[0]):
        i = labels.index(value)
        B[i][j] = 1.

    return B


def rand(rows: int, cols: int) -> Tensor2D:
    return [[random.uniform(-0.5, 0.5) for i in range(cols)] for j in range(rows)]


def shape(A: Tensor2D) -> Shape2D:
    rows = len(A)
    cols = len(A[0])  # assume all rows are equal length
    return rows, cols


def sum_rows(A: Tensor2D) -> Tensor2D:
    return [[sum(row)] for row in A]


def sum_cols(A: Tensor2D) -> Tensor2D:
    A_shape = shape(A)
    B = zeros(1, A_shape[1])

    for j in range(A_shape[1]):
        for i in range(A_shape[0]):
            B[0][j] += A[i][j]

    return B


def sum_all(A: Tensor2D) -> float:
    return sum(sum(row) for row in A)


def transpose(A: Tensor2D) -> Tensor2D:
    A_shape = shape(A)
    A_transposed = zeros(*reversed(A_shape))
    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            A_transposed[j][i] = A[i][j]
    return A_transposed


def zeros(rows: int, cols: int) -> Tensor2D:
    return [[0. for i in range(cols)] for j in range(rows)]
