import tensorflow.keras.backend as K
print("Create Generators")
from shared.generators.generators import get_test_gen
print("Assemble & Compile Model")
import sys
import importlib
rundef_path = sys.argv[1]
rundef = importlib.import_module(rundef_path)  # Run definition
import shared.definitions.paths as paths

"""Model Loading"""
checkpoint_dir = paths.models + 'alexnet/outputs/checkpoints_v20/'
model_file = 'weights.58-2.73.hdf5'

print("Loading Model & Generator")
rundef.model.load_weights(checkpoint_dir + model_file)
test_gen = get_test_gen(rundef.test_batch_size, rundef.num_classes)

preds = rundef.model.predict_generator(test_gen, max_queue_size=4, workers=4, verbose=1)

repeats = 10
mean_preds = None

for n in range(repeats):
    if n == 0:
        mean_preds = preds[n::repeats]
    else:
        mean_preds += preds[n::repeats]
mean_preds = K.constant(mean_preds/repeats)
mean_labels = K.constant(test_gen.y_labels)

print(mean_preds.get_shape().as_list())
print(mean_labels.get_shape().as_list())

acc1 = rundef.top_1_acc(mean_labels, mean_preds)
acc5 = rundef.top_5_acc(mean_labels, mean_preds)

acc1_f = K.sum(acc1)/acc1.get_shape().as_list()[0]
acc5_f = K.sum(acc5)/acc5.get_shape().as_list()[0]

# v24: 0.5913 0.81508, seems to at most gain 0.0001 and 0.001 top-1 and top-5 accuracy with eigenvector shift
# v23: 0.58926 0.81532, but had much higher loss... thinking lower loss doesn't matter much then
# v22: 0.56186 0.79306, lower, but the accuracies in Tensorboard were lower too so no surprise there
# v20: 0.58338 0.8142, reducing LRN coefficient seems to have made a but of a difference in generalization then
tf_session = K.get_session()
print(acc1_f.eval(session=tf_session), acc5_f.eval(session=tf_session))