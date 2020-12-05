from deriv8.matrix2d import Tensor2D, negate, element_multiply_log, divide


# Calculate the negative log of predicated probability of correct answer.
#    -log(y_hat) where y = 1
# Y is the one-hot 0/1 for each class, where only one class is one and the rest are zero.
# Y_hat is the softmax generated probability 0.0 (0%) to 1.0 (100%). We take a log of that.
# Multiplying these together will leave us with just the probability for the correct answer.
def loss(Y_hat, Y: Tensor2D) -> Tensor2D:
    # Note: element_multiply_log skips evaluating log(y_hat) when y is zero.
    return negate(element_multiply_log(Y, Y_hat))
