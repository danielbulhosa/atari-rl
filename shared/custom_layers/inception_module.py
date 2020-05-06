import keras.layers as lyr


def inception(input, one, three_red, three, five_red, five, pool_proj,
              kernel_initializer, bias_initializer,
              kernel_regularizer, bias_regularizer):
    """
    Create an inception module with the specified parameters and
    apply it to the input.

    Note that per the paper all convolutions within the inception
    module use ReLu activations. Also note we use the same initialization
    and regularization for all layers in the module.

    :param input: The tensor input into the inception module.
    :param one: The number of 1x1 convolutional filters.
    :param three_red: The number of 1x1 convolutional filters used to compress the 3x3 filter input.
    :param three: The number of 3x3 filters.
    :param five_red: The number of 1x1 convolutional filters used to compress the 5x5 filter input.
    :param five: The number of 5x5 filters.
    :param pool_proj: The number of 1x1 filters used to compress the 3x3 max pooling output.
    :param kernel_initializer: Kernel initializer for all layers in module.
    :param bias_initializer: Bias initializer for all layers in module.
    :param kernel_regularizer: Kernel regularizer for all layers in module.
    :param bias_regularizer: Bias regularizer for all layers in module.
    :return: The tensor representing the concatenation of all output feature maps.
    """

    def get_convolution(filters, size, strides):
        return lyr.Conv2D(filters, size, strides=strides, padding='same',
                          activation='relu',
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer
                          )

    one_out = get_convolution(one, 1, 1)(input)

    three_intermediate = get_convolution(three_red, 1, 1)(input)

    three_out = get_convolution(three, 3, 1)(three_intermediate)

    five_intermediate = get_convolution(five_red, 1, 1)(input)

    five_out = get_convolution(five, 5, 1)(five_intermediate)

    pool_proj_intermediate = lyr.MaxPool2D(3, 1, 'same')(input)

    pool_proj_out = get_convolution(pool_proj, 1, 1)(pool_proj_intermediate)

    output = lyr.Concatenate(axis=-1)([one_out, three_out, five_out, pool_proj_out])

    return output
