from deriv8.matrix2d import Matrix2D, element_exp, minus, shape, zeros


def softmax(Z: Matrix2D, stable=True) -> Matrix2D:
    Z_shape = shape(Z)

    if stable:
        # stable softmax via https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        Z_max = max(Z[0])
        Z_minus_max = minus(Z, [[Z_max]])
        Z_exp = element_exp(Z_minus_max)
    else:
        Z_exp = element_exp(Z)

    Z_exp_col_sum = zeros(1, Z_shape[1])

    for i in range(Z_shape[0]):
        for j in range(Z_shape[1]):
            Z_exp_col_sum[0][j] += Z_exp[i][j]

    Z_softmax = zeros(*Z_shape)
    for i in range(Z_shape[0]):
        for j in range(Z_shape[1]):
            Z_softmax[i][j] = Z_exp[i][j] / Z_exp_col_sum[0][j]

    return Z_softmax


def softmax_derivative(A: Matrix2D) -> Matrix2D:
    A_shape = shape(A)
    N = A_shape[1]
    dZ_dA = zeros(N, N)

    for j in range(N):
        for i in range(N):
            if i == j:
                dZ_dA[i][j] = A[0][i] * (1. - A[0][j])
            elif i > j:
                dZ_dA[i][j] = A[0][i] * (-A[0][j])
            else:
                # resulting matrix is mirrored diagonally, so we just copy the value (saves a multiplication)
                dZ_dA[i][j] = dZ_dA[j][i]

    return dZ_dA