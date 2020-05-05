import keras.layers as lyr


def residual_unit(input, filters, downsample, kernel_initializer, bias_initializer,
                  kernel_regularizer, bias_regularizer):
    """
    Residual unit using pre-activation as described in:
    https://arxiv.org/pdf/1603.05027v2.pdf

    Note that we use 1x1 convolutions to transform the
    dimension of residuals so we can increase filters
    and downsample spatial dimensions.

    Ideally we would do zero-padding along filter axis
    and downsample through stride 2 or some type of pooling.
    However, this would require more code complexity (while
    reducing parameter complexity). We may implement a lambda
    layer doing exactly this later on.

    :param input: The input tensor.
    :param filters: The number of filters in each convolutional layer of the residual unit.
    :param downsample: Whether to downsample at the beginning of the layer. If so we downsample by 2
                       and we use 1x1 convolutions to resize the residual.
    :param kernel_initializer: Kernel initializer for all layers in module.
    :param bias_initializer: Bias initializer for all layers in module.
    :param kernel_regularizer: Kernel regularizer for all layers in module.
    :param bias_regularizer: Bias regularizer for all layers in module.
    :return: The output of the residual unit, which consists of the sum of the output of the
             previous layer and the output of the layers in the residual unit.
    """

    strides = 2 if downsample else 1

    def get_convolution(filters, size):
        return lyr.Conv2D(filters, size, strides=strides, padding='same',
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer
                          )

    int1 = lyr.BatchNormalization()(input)
    int2 = lyr.ReLU()(int1)
    int3 = get_convolution(filters, 3)(int2)
    int4 = lyr.BatchNormalization()(int3)
    int5 = lyr.ReLU()(int4)
    int6 = get_convolution(filters, 3)(int5)

    # If downsampling we use convolutional filters to increase filters
    # and reduce the size of the image. This gets dimensions to match.
    if downsample:
        res = get_convolution(filters, 1)
    else:
        res = input

    out = lyr.add([res, int6])

    return out
