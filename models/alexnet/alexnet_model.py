import tensorflow as tf
import keras.models as mod
import keras.layers as lyr
import keras.regularizers as reg
import keras.initializers as init
import keras.optimizers as opt
import keras.metrics as met
import keras.losses as losses
from shared.custom_layers.local_response_normalization import lrn_parametric, lrn_shape

k, n, alpha, beta = 2, 5, 1, 0.75
lrn = lambda tensor: lrn_parametric(tensor, k, n, alpha, beta)

alexnet = mod.Sequential(
 [# We input 224 x 224 pixel images with 3 channels (rgb)
  # We create 96 feature maps, by using an 11x11 pixel kernel with stride 4
  lyr.Conv2D(96, 11, strides=4, padding='same',
             input_shape=(224, 224, 3,), activation='relu',
             kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(),
             kernel_regularizer=reg.l2(0.0005),
             bias_regularizer=reg.l2(0.0005)
             ),
  # The lambdas here are the local response normalization
  lyr.Lambda(lrn, output_shape=lrn_shape),
  lyr.MaxPool2D(3, 2),
  lyr.Conv2D(256, 5, strides=1, padding='same', activation='relu',
             kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(), # original: init.Ones()
             kernel_regularizer=reg.l2(0.0005),
             bias_regularizer=reg.l2(0.0005)
             ),
  lyr.Lambda(lrn, output_shape=lrn_shape),
  lyr.MaxPool2D(3, 2),
  lyr.Conv2D(384, 3, strides=1, padding='same', activation='relu',
             kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(),
             kernel_regularizer=reg.l2(0.0005),
             bias_regularizer=reg.l2(0.0005)
             ),
  lyr.Conv2D(384, 3, strides=1, padding='same', activation='relu',
             kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(), # original: init.Ones()
             kernel_regularizer=reg.l2(0.0005),
             bias_regularizer=reg.l2(0.0005)
             ),
  lyr.Conv2D(256, 3, strides=1, padding='same', activation='relu',
             kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(), # original: init.Ones()
             kernel_regularizer=reg.l2(0.0005),
             bias_regularizer=reg.l2(0.0005)
             ),
  # Note from the diagram there is one last pool op after the last conv
  lyr.MaxPool2D(3,2),
  lyr.Flatten(),
  lyr.Dense(4096, activation='relu',
            kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
            bias_initializer=init.Zeros(),  # original: init.Ones()
            kernel_regularizer=reg.l2(0.0005),
            bias_regularizer=reg.l2(0.0005)
            ),
  # The paper says the first two dense layers are regularized with dropout
  lyr.Dropout(0.5),
  lyr.Dense(4096, activation='relu',
            kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
            bias_initializer=init.Zeros(),  # original: init.Ones()
            kernel_regularizer=reg.l2(0.0005),
            bias_regularizer=reg.l2(0.0005)
            ),
  lyr.Dropout(0.5),
  lyr.Dense(1000,
            activation='softmax',
            kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
            bias_initializer=init.Zeros(),  # original: init.Ones()
            kernel_regularizer=reg.l2(0.0005),
            bias_regularizer=reg.l2(0.0005)
            ),
 ]
)

# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()

"""
Using zero built in decay and relying exclusively on the heuristic
used in the original paper.
"""
optimizer = opt.Adam(0.0001)
# This line below would match the paper exactly but was MUCH slower than using Adam on 2 class example
# optimizer = opt.SGD(learning_rate=0.01, momentum=0.9, decay=0.0) # Note decay refers to learning rate

top_1_acc = met.categorical_accuracy


def loss(y_true, y_pred):
    return losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)


def top_5_acc(y_true, y_pred):
    return  met.top_k_categorical_accuracy(y_true, tf.cast(y_pred, dtype='float32'), k=5)


alexnet.compile(optimizer=optimizer,
                loss=loss,
                metrics=[top_1_acc, top_5_acc],
                # options=run_options, FIXME: Causes problems in Pycharm
                # run_metadata=run_metadata
                )

if __name__ == '__main__':
    print(alexnet.summary())