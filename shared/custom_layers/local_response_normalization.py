import numpy as np
from keras import backend as K
import tensorflow as tf


def lrn_parametric(tensor, k, n, alpha, beta):
    """
    Check that parameters make sense (note n/2 describes
    the number of adjacent layers on each side to normalize by)
    """
    assert n >= 1  # n < 0 doesn't make sense, n = 0 (when k = 0) makes the layer output constant
    assert k > 0  # If k = 0 we can get div by zero errors or numeric instability
    assert alpha > 0  # alpha = 0 is equivalent to reducing the layer to scalar division
    assert beta > 0  # beta = 0 is equivalent to removing this layer

    tf_session = K.get_session()

    """
    First dimension is batches, second and third are spatial, 
    fourth should be layers
    """
    shape = tensor.get_shape().as_list()
    assert len(shape) == 4  # Make sure shape is (batch, x, y, map)
    N = shape[3]

    """
    For layer i only the layers i +/- int_step fall within 
    the range defined by the sum formula in the paper
    """
    int_step = n / 2 - (n / 2 % 1)

    """
    These tensors correspond to i, the sum lower bounds, 
    and upper bounds respectively
    """
    i = np.arange(N)
    j_lower = np.maximum(0, i - int_step).reshape(-1, 1)
    j_upper = np.minimum(N - 1, i + int_step).reshape(-1, 1)

    """
    Create a tensor Mij such that Mij = 1 if j_lower(i) <= j <= j_upper(i) and is 0 otherwise
    If a_{xy}^j is the convolution output then M_{ij}a_{xy}^j gives the 
    lrn normalization coefficient
    """
    mask_ij = np.repeat(np.arange(N).reshape(1, -1), N, axis=0)
    mask_ij = np.logical_and(mask_ij >= j_lower, mask_ij <= j_upper).astype(int)
    mask_ij = K.variable(mask_ij)

    """
    Calculate the  norm tensor, denominator, and finally 
    output tensor of LRN. Note that even because K is a Keras 
    wrapper around Tensorflow, we can use tf methods on tensors 
    created with K. This is because K generated objects are 
    really tf objects
    """
    lrn_norm = tf.einsum('bxyj,ij->bxyi', K.pow(tensor, 2), mask_ij)
    denom = K.pow(k + alpha * lrn_norm, beta)
    lrn_output = tensor / denom

    return lrn_output


def lrn_shape(input_shape):
    return input_shape


if __name__ == '__main__':
    k, n, alpha, beta = 1, 9, 1, 1
    test_array = np.ones((1, 1, 1, 20))
    print("\nTest layer functionality...")
    # correct, window at edges and normalization makes sense: divide each one by 9 ones + a constant one (i.e. 10)
    print(K.eval(lrn_parametric(K.constant(test_array), k, n, alpha, beta)))
    # correct, square of all results above
    print(K.eval(lrn_parametric(K.constant(test_array), k, n, alpha, 2)))
    # correct, double of all first results (because of numerator), factors of 2 in denominator cancel
    print(K.eval(lrn_parametric(K.constant(2 * test_array), k, n, 1/4, beta)))
    # correct, still dividing by the same total (5 constant 1's and 5 variable 1's) but window at edges smaller
    print(K.eval(lrn_parametric(K.constant(test_array), 5, 5, alpha, beta)))
    # correct, dimensions are as expected and normalization only along last axis (which is feature axis in our network)
    print(K.eval(lrn_parametric(K.constant(np.ones((2, 2, 2, 20))), k, n, alpha, beta)))

    # test parameter sensitivity
    print("\nTest parameter sensitivity...")
    k, n, alpha, beta = 0.2, 5, 0.1, 0.75
    test_array = np.array([[[[0.1]*24 + [2] + [0.1]*24 + [1] + [0.1]*24 + [-1] + [-0.1]*24 ]]])
    print(K.eval(K.constant(test_array)))
    print(K.eval(lrn_parametric(K.constant(test_array), 2, n, 1, beta)))
    print(K.eval(lrn_parametric(K.constant(test_array), k, n, alpha, beta)))
    print(K.eval(lrn_parametric(K.constant(test_array), 0.02, n, 0.01, beta)))