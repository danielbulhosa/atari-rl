import keras.backend as K # FIXME- FP16 makes accuracy measures disappear, doesn't seem to sped things up...
#dtype='float16' # FIXME remove F16 accuracy??
#K.set_floatx(dtype)
#K.set_epsilon(1e-4) # Default is 1e-7 which is too small for FP16

# Set memory restrictions to allow some room for our monitor to render :)
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
session = tf.Session(config=config)
K.set_session(session=session)

import keras.callbacks as call
import gc
print("Create Generators")
from shared.generators.generators import train_gen, val_gen
print("Assemble & Compile Model")
from models.alexnet.alexnet_model_dev import alexnet  #, run_metadata FIXME: Causes problems in Pycharm, changed to `alexnet_model` import
from tensorflow.python.client import timeline

"""Model Loading (If Applicable)"""
checkpoint_dir = 'alexnet/checkpoints_v27/'
model_file = None
epoch_start = None

if model_file is not None and checkpoint_dir is not None:
    print("Loading Model")
    alexnet.load_weights(checkpoint_dir + model_file)
    epochs = 90 - epoch_start
else:
    epoch_start = 0
    epochs = 90

"""Callbacks"""
# FIXME CHANGE #11 - Divide learning rate by 2
scheduler = call.ReduceLROnPlateau(factor=0.1, monitor='val_categorical_accuracy', verbose=1, mode='auto',
                                   patience=5, min_lr=10**(-8), min_delta=0.0001)  # FIXME - change back to monitoring loss?
tensorboard = call.TensorBoard(log_dir='alexnet/logs_v27',  # FIXME -- update dir
                               batch_size=128,
                               write_grads=True,
                               write_images=True)
checkpointer = call.ModelCheckpoint('alexnet/checkpoints_v27/weights.{epoch:02d}-{val_loss:.2f}.hdf5',  # FIXME -- update dir
                                   verbose=1)

# Added to address OOM Error - not sure if needed anymore
# See: https://github.com/keras-team/keras/issues/3675
garbage_collection = call.LambdaCallback(on_epoch_end=lambda epoch, logs: gc.collect())

""" Model train code """
print("Begin Training Model")
alexnet.fit_generator(train_gen,
                      epochs=90,  # FIXME: make 90, reduce for debugging
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
