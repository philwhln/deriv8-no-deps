from deriv8.matrix2d import Tensor2D, element_exp, minus, shape, zeros


def softmax(Z: Tensor2D, stable=True) -> Tensor2D:
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
