import keras.models as mod
import keras.layers as lyr
import keras.optimizers as opt
import keras.metrics as met
import keras.callbacks as call
import tensorflow as tf
import keras.losses as losses
import keras.regularizers as reg
import keras.initializers as init
import shared.definitions.paths as paths
import os
from os import path

"""
Model Definition
"""
version = 0
num_classes = 1000  # FIXME - what of this do we need anymore?
num_datapoints = None
steps_per_epoch = None

# FIXME - how common is it for RL agents to use regularization like this?
init_reg = {
    'kernel_initializer': init.he_uniform(),
    'bias_initializer': init.Zeros(),
    'kernel_regularizer': reg.l2(0),
    'bias_regularizer': reg.l2(0)
}

# FIXME - add model used for agent in question
out = None

model = mod.Model(inputs=input, outputs=out)



"""
Optimizer, Loss, & Metrics

Using zero built in decay and relying exclusively on the heuristic
used in the original paper.
"""
optimizer = opt.Adam(0.001)  # FIXME - do we want to change our optimizer?


def loss(y_true, y_pred):
    return losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)


model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[None],  # FIXME - what metrics do we use?
              )

"""
Epochs & Batch Sizes
"""
# FIXME - RL agents operate not on epochs but on iterations and episodes. How do we capture this?
num_epochs = 90
train_batch_size = 256
val_batch_size = 256
test_batch_size = 26

"""
Callback Params
"""

# FIXME - Need to use learning rate scheduler used in different papers?
scheduler = None

log_dir = paths.agents + 'dqn/v{:02d}/logs'.format(version)
checkpoint_dir = paths.agents + 'dqn/v{:02d}/checkpoints'.format(version)

if not path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

if not path.isdir(log_dir):
    os.mkdir(log_dir)

# Experiment directory format is {model}/{version}/{filetype}
tensorboard_params = {'log_dir': log_dir,
                      'batch_size': train_batch_size,
                      'write_grads': True,
                      'write_images': True}

checkpointer_params = {'filepath': checkpoint_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                       'verbose': 1}

"""
Loading Params
"""

# If model_file is not None and checkpoint_dir is not None then loading happens, else not
loading_params = {'checkpoint_dir': None,
                  'model_file': None,
                  'epoch_start': None}


if __name__ == '__main__':
    print(model.summary())