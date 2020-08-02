from keras import backend as K
import tensorflow as tf
from keras import layers as lyr
import numpy as np


def sample_weights_and_gradients(model, train_gen):
    """
    Takes a model and a minibatch of training data
    and prints out weights, gradients, and loss tensors.

    Useful for debugging silent gradient and weight problems.

    :param model: The model to debug.
    :param train_gen: The generator used for getting training data.
    """

    """Check that gradients propagate correctly with indexing"""
    # an input layer to feed labels
    y_true = lyr.Input((1, ))
    # compute loss based on model's output and true labels
    diff = tf.reshape(model.output[1], (-1, 1)) - y_true
    se = K.square(diff)
    mse = K.mean(se, axis=-1)
    # get a single minibatch
    x, y = train_gen[0]
    y_pred = model.predict(x)[1]

    print("\nInput States & Actions")
    print(x)
    print("\nInput Target Values")
    print(y)
    print("\nModel Predictions (Manually Calculated)")
    print(y_pred)
    print("\nPrediction-Target Difference (Manually Calculated)")
    print(y_pred-y)

    with K.get_session() as sess:

        print("\nSquared Errors (Manually Calculated)")
        print(sess.run(K.square(y_pred.reshape(-1, 1)-y.reshape(-1, 1))))
        print("\nMean Squared Error (Manually Calculated):")
        print(sess.run(K.mean(K.square(y.reshape(-1, 1)-y_pred.reshape(-1, 1)), axis=-1)))

        print("\nMinibatch size: {}".format(len(x)))

        sess.run(tf.initialize_all_variables())
        evaluated_out = sess.run(model.output[1], feed_dict={model.input[0]: x[0],
                                                             model.input[1]: x[1].reshape(-1, 1),
                                                             y_true: y.reshape(-1, 1)})
        evaluated_diff = sess.run(diff, feed_dict={model.input[0]: x[0],
                                                   model.input[1]: x[1].reshape(-1, 1),
                                                   y_true: y.reshape(-1, 1)})
        evaluated_se = sess.run(se, feed_dict={model.input[0]: x[0],
                                               model.input[1]: x[1].reshape(-1, 1),
                                               y_true: y.reshape(-1, 1)})
        evaluated_mse = sess.run(mse, feed_dict={model.input[0]: x[0],
                                                 model.input[1]: x[1].reshape(-1, 1),
                                                 y_true: y.reshape(-1, 1)})

        print("\nOutputs Shape (Calculated From Model): {}".format(evaluated_out.shape))
        print("\nOutputs:")
        print(evaluated_out)
        print("\nDifference Shape (Calculated From Model): {}".format(evaluated_diff.shape))
        print("\nDifferences:")
        print(evaluated_diff)
        print("\nSquared Errors Shape (Calculated From Model): {}".format(evaluated_se.shape))
        print("\nSquared Errors:")
        print(evaluated_se)
        print("\nMean Squared Errors Shape (Calculated From Model): {}".format(evaluated_mse.shape))
        print("\nMean Squared Errors:")
        print(evaluated_mse)

        for layer in model.layers:
            # Skip indexing input layer
            if layer.dtype == 'int32':
                continue
            # compute gradient of loss with respect to inputs
            # For indexing explanation: https://stackoverflow.com/questions/49834380/k-gradientsloss-input-img0-return-none-keras-cnn-visualization-with-ten
            grad_mse = K.gradients(mse, layer.output)[0]

            evaluated_gradients = sess.run(grad_mse, feed_dict={model.input[0]: x[0],
                                                                model.input[1]: x[1].reshape(-1, 1),
                                                                y_true: np.array(y).reshape(-1, 1)})

            print('\n\n')
            print(layer.name)
            weight_list = layer.get_weights()
            weights = None if len(weight_list) == 0 else weight_list[0]
            weights_shape = None if weights is None else weights.shape
            print("\nWeight tensor shape (Calculated From Model): {}".format(weights_shape))
            print("Weight Tensor:")
            print(weights)
            print("\nGradient shape (Calculated From Model): {}".format(evaluated_gradients.shape))
            print("Gradient Tensor:")
            print(evaluated_gradients)


def test_generator(train_gen, max_epoch=None):
    """
    Runs training generator for defined number of epochs
    to check no minibatches raise errors.

    :param train_gen: The generator creating training data.
    """

    epochs = train_gen.epoch if max_epoch is None else max_epoch + 1

    print("\n")
    for epoch in range(1, epochs):
        break  # FIXME, remove break
        print("Generator Test: Epoch {}".format(epoch))
        train_gen[epoch]
