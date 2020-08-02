import keras.models as mod
import keras.layers as lyr
import keras.optimizers as opt
import keras.metrics as met
import keras.losses as losses
import keras.initializers as init
import tensorflow as tf
import shared.definitions.paths as paths
import os
from os import path
from shared.generators.environment_sequence import EnvironmentSequence
import shared.debugging.functions as debug
import gym
import copy

"""
Model Definition
"""
version = 0
num_datapoints = None
steps_per_epoch = None


def get_Q_a(tensors):
    import tensorflow as tf  # FIXME - janky, need a better solution
    return tf.reshape(tf.gather_nd(tensors[0], tensors[1], batch_dims=1), (-1, 1))


def Q_a_shape(input_shape):
    return input_shape[1]


kernel_init = init.RandomUniform(-0.1, 0.1)

input_size = 4
output_size = 2  # Same as number of actions
state_input = lyr.Input((input_size, ))
action_input = lyr.Input((1, ), dtype='int32')
intermediate1 = lyr.Dense(24, activation='relu', kernel_initializer=kernel_init)(state_input)
intermediate2 = lyr.Dense(24, activation='relu', kernel_initializer=kernel_init)(intermediate1)
value_out = lyr.Dense(output_size, activation='linear', kernel_initializer=kernel_init)(intermediate2)
# Note we use the action to index for the value used for the loss. It is NOT used as a feature for training,
action_out = lyr.Lambda(get_Q_a, Q_a_shape)([value_out, action_input])
model = mod.Model(inputs=[state_input, action_input], outputs=[value_out, action_out])

optimizer = opt.RMSprop(0.001)  # FIXME - do we want to change our optimizer?

model._make_predict_function()
graph = tf.get_default_graph()
model.compile(optimizer=optimizer,
              loss=[None, losses.mean_squared_error],
              metrics=[met.mean_squared_error],
              loss_weights=[0, 1]
              )

"""
Epochs & Batch Sizes
"""
# We fixed the number of iterations that constitute an epoch in the generator,
# Note that we do not need a validation generator hence not val batch size.
num_epochs = 90
environment = gym.make("CartPole-v1")
train_exploration_schedule = (lambda iteration: 0.1)
eval_exploration_schedule = (lambda iteration: 0.05)
grad_update_frequency = 1
train_batch_size = 512
target_update_frequency = None  # Multiplying by grad update freq lets us interpret the second number as epochs
action_repeat = 1  # The average episode length at the beginning is 9 steps, 4 repetitions would be half the episode...
gamma = 1
epoch_length = 1000
replay_buffer_size = 16 * 1000  # Multiplying by grad update freq lets us interpret the second number as epochs
replay_buffer_min = 10000
eval_episodes = 100
eval_num_samples = 1000

train_gen = EnvironmentSequence(model,
                                source_type='value',
                                environment=copy.deepcopy(environment),
                                graph=graph,
                                epsilon=train_exploration_schedule,
                                batch_size=train_batch_size,
                                grad_update_frequency=grad_update_frequency,
                                target_update_frequency=target_update_frequency,
                                action_repeat=action_repeat,
                                gamma=gamma,
                                epoch_length=epoch_length,
                                replay_buffer_size=replay_buffer_size,
                                replay_buffer_min=replay_buffer_min)

"""
Callback Params
"""

log_dir = paths.agents + 'cartpole_v1/v{:02d}/logs'.format(version)
checkpoint_dir = paths.agents + 'cartpole_v1/v{:02d}/checkpoints'.format(version)

if not path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

if not path.isdir(log_dir):
    os.mkdir(log_dir)

"""
Callback and Generator Shared Parameters
"""


# Experiment directory format is {model}/{version}/{filetype}
tensorboard_params = {'log_dir': log_dir,
                      'batch_size': train_batch_size,
                      'write_grads': True,
                      'write_images': True}

checkpointer_params = {'filepath': checkpoint_dir + '/weights.{epoch:02d}.hdf5',
                       'verbose': 1}

evaluator_params = {'sequence_constructor': train_gen.create_validation_instance,
                    'epsilon': eval_exploration_schedule,
                    'init_state_dir': None,
                    'num_episodes': eval_episodes,
                    'num_init_samples': eval_num_samples}

"""
Loading Params
"""

# If model_file is not None and checkpoint_dir is not None then loading happens, else not
loading_params = {'checkpoint_dir': None,
                  'model_file': None,
                  'test_checkpoint_dir': checkpoint_dir,
                  'test_model_file': '/weights.{epoch:02d}.hdf5',  # Weight file used for testing performance
                  'test_model_dir': os.path.dirname(os.path.realpath(__file__)),
                  'epoch_start': None}

if __name__ == '__main__':
    model.summary()
    debug.sample_weights_and_gradients(model, train_gen)
    debug.test_generator(train_gen)
