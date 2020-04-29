i# Set memory restrictions to allow some room for our monitor to render :)
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
session = tf.Session(config=config)
K.set_session(session=session)

import keras.callbacks as call
import gc
from shared.generators.generators import get_train_gen, get_val_gen
print("Assemble & Compile Model")
from models.alexnet.alexnet_model_dev import alexnet, scheduler_params, tensorboard_params, checkpointer_params,\
                                             loading_params, num_epochs, train_batch_size, val_batch_size,\
                                             shift_scale, aug_list

"""Model Loading (If Applicable)"""
checkpoint_dir = loading_params['checkpoint_dir']
model_file = loading_params['model_file']
epoch_start = loading_params['epoch_start']

if model_file is not None and checkpoint_dir is not None:
    print("Loading Model")
    alexnet.load_weights(checkpoint_dir + model_file)
    epochs = num_epochs - epoch_start
else:
    epoch_start = 0
    epochs = num_epochs

"""Callbacks"""
scheduler = call.ReduceLROnPlateau(**scheduler_params)
tensorboard = call.TensorBoard(**tensorboard_params)
checkpointer = call.ModelCheckpoint(**checkpointer_params)

# Added to address OOM Error - not sure if needed anymore
# See: https://github.com/keras-team/keras/issues/3675
garbage_collection = call.LambdaCallback(on_epoch_end=lambda epoch, logs: gc.collect())

print("Create Generators")
"""Generators"""
train_gen = get_train_gen(train_batch_size, shift_scale, aug_list)
val_gen = get_val_gen(val_batch_size)


""" Model train code """
print("Begin Training Model")
alexnet.fit_generator(train_gen,
                      epochs=num_epochs,
                      validation_data=val_gen,
                      verbose=1,  # 0 in notebook, verbose doesn't slow down training, we checked
                      callbacks=[tensorboard,
                                 scheduler,
                                 checkpointer,
                                 garbage_collection,
                                 ],
                      shuffle=True, # Note this only works because we removed step parameters
                      use_multiprocessing=False, # Actually significantly faster when this is false.
                      workers=4,  # Optimal: More workers doesn't increase images/second, makes it slightly slower
                      max_queue_size=4,  # Optimal: A larger queue seems to not make much of a difference
                      initial_epoch=epoch_start
                     )
