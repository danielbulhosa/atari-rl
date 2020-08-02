import keras.models as mod
import keras.layers as lyr
import keras.optimizers as opt
import keras.metrics as met
import keras.losses as losses
import keras.initializers as init
import shared.definitions.paths as paths
import os
from os import path
from shared.generators.atari_sequence import AtariSequence
import shared.debugging.functions as debug
import gym
import copy
import math
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


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


# As far as we can find there is no weight decay specified on the paper. We use a reasonable initialization then.
init_params = {'kernel_initializer': init.glorot_uniform()}

# Note we don't need to set repeat_action_probability (ALE level) because repeat now handled at Gym package level
# using `frameskip` parameter. See https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
builtin_action_repeat = (3, 5)
builtin_action_repeat_avg = builtin_action_repeat if isinstance(builtin_action_repeat, int) \
                            else (builtin_action_repeat[1] - builtin_action_repeat[0])/2 + builtin_action_repeat[0]
environment = gym.make("Breakout-v0", frameskip=builtin_action_repeat)
input_image_dims = (84, 84)
num_stack = 4
output_size = environment.action_space.n  # Same as number of actions

state_input = lyr.Input((input_image_dims[0], input_image_dims[1], num_stack))
action_input = lyr.Input((1, ), dtype='int32')
intermediate1 = lyr.Conv2D(32, 8, strides=4, padding='same', activation='relu', **init_params)(state_input)
intermediate2 = lyr.Conv2D(64, 4, strides=2, padding='same', activation='relu', **init_params)(intermediate1)
intermediate3 = lyr.Conv2D(64, 3, strides=1, padding='same', activation='relu', **init_params)(intermediate2)
intermediate4 = lyr.Flatten()(intermediate3)
intermediate5 = lyr.Dense(512, activation='relu', **init_params)(intermediate4)
value_out = lyr.Dense(output_size, activation='linear', **init_params)(intermediate5)
# Note we use the action to index for the value used for the loss. It is NOT used as a feature for training,
action_out = lyr.Lambda(get_Q_a, Q_a_shape)([value_out, action_input])
model = mod.Model(inputs=[state_input, action_input], outputs=[value_out, action_out])

# These are the parameters for the optimizer specified in the paper. Apparently Adam = RMSprop + Momentum
optimizer = opt.Adam(0.00025, beta_1=0.95, beta_2=0.95, epsilon=0.01, clipnorm=1)

model._make_predict_function()
graph = tf.get_default_graph()
model.compile(optimizer=optimizer,
              loss=[None, losses.mean_squared_error],
              metrics=[met.mean_squared_error],
              loss_weights=[0, 1]
              )

"""
Epochs & Batch Sizes

IMPORTANT: Each iteration corresponds to # `builtin_action_repeat_avg` frames on average (4 in our case)
Each weight update in turn corresponds to `grad_update_frequency` iterations.

We need these different measures because minibatch updates are counted in weight updates,
simulation calls are counted in iterations, but the natural scale of the problem itself 
is frames.
"""

grad_update_frequency = 4  # Measured in iterations. From paper, 1 grad update = 4 actions = 16 frames
final_exploration_iteration = int(10**6 / builtin_action_repeat_avg)  # Measured in frames, converted to iterations
train_exploration_schedule = (lambda iteration: max(0.1, 1 - 0.9 * iteration/final_exploration_iteration))
eval_exploration_schedule = (lambda iteration: 0.05)
train_batch_size = 32
target_update_frequency = int(10**4 * grad_update_frequency)  # Measured in weight updates, converted to iterations
gamma = 0.99
epoch_length = 50000  # Measured in weight updates, used by generator getter
epoch_iterations = epoch_length * grad_update_frequency  # Convert epoch length from weight updates to iterations
total_training_iterations = 200 * 10**6 // builtin_action_repeat_avg  # Measured in frames, converted to iterations
num_epochs = int(math.ceil(total_training_iterations / epoch_iterations))
replay_buffer_size = 10**6  # Note that the paper measures this number in frames
replay_buffer_min = 50 * 10**3  # Measured in frames STORED IN BUFFER, technically more frames are played due to repeats
eval_episodes = np.inf  # No upper bound on the number of episodes evaluated
eval_max_iter = 10000 // builtin_action_repeat_avg  # Upper bound of evaluation measured in frames, converted to iters
eval_num_samples = 10000
take_pair_max = True

"""
Training & Validation Generators
"""

train_gen = AtariSequence(model,
                          source_type='value',
                          environment=copy.deepcopy(environment),
                          graph=graph,
                          n_stack=num_stack,
                          stack_dims=input_image_dims,
                          pair_max=True,  # We checked and the underlying simulator does NOT take pairwise max
                          epsilon=train_exploration_schedule,
                          batch_size=train_batch_size,
                          grad_update_frequency=grad_update_frequency,
                          target_update_frequency=target_update_frequency,
                          action_repeat=1,  # No need for manual action repeat, Gym environment handles repeats
                          gamma=gamma,
                          epoch_length=epoch_length,
                          replay_buffer_size=replay_buffer_size,
                          replay_buffer_min=replay_buffer_min,
                          use_double_dqn=True,
                          )

"""
Callback Params
"""

# Note: There is no learning rate schedule in the papers, it's constant

log_dir = paths.agents + 'breakout_v0/v{:02d}/logs'.format(version)
checkpoint_dir = paths.agents + 'breakout_v0/v{:02d}/checkpoints'.format(version)
data_dir = paths.agents + 'breakout_v0/v{:02d}/data'.format(version)

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

evaluator_params = {'num_episodes': eval_episodes,
                    'num_max_iter': eval_max_iter,
                    'num_init_samples': eval_num_samples,
                    'sequence_constructor': train_gen.create_validation_instance,
                    'epsilon': eval_exploration_schedule,
                    'init_state_dir': data_dir}

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
    debug.test_generator(train_gen, 1)
