import cupy as cp
import os
import time
import re
import numpy as np
import pickle
from shared.generators.imagenet_sequence import AlexNetSequence
import shared.definitions.paths as paths

"""Load pixel averages"""
results = None
with open(paths.constants + 'pixel_avg.pkl', 'rb') as file:
    results = pickle.load(file)

pixel_avg = results['pixel_avg']

"""Load pixel covariances"""
# The covariances roughly align with the precalculated values used here
# https://medium.com/@kushajreal/training-alexnet-with-tips-and-checks-on-how-to-train-cnns-practical-cnns-in-pytorch-1-61daa679c74a
cov_results = None
with open(paths.constants + 'cov.pkl', 'rb') as file:
    cov_results = pickle.load(file)

cov = cov_results['cov']
# Note that the eigenvectors returned are normalized
eigenvalues, eigenvectors = cp.linalg.eigh(cov/255**2)
stdev = cp.sqrt(cov.diagonal()/255**2)

"""Map Classes To Names & One-Hot Encodings"""
with open(paths.dataset + 'sysnet_words.txt', 'r') as file:
    words = file.readlines()

codes = [re.match('n[0-9]*', word).group() for word in words]
labels = [word.replace(code, '').strip() for word, code in zip(words, codes)]
code_label_map = dict(zip(codes, labels))
index_code_map = dict(enumerate(codes))
code_index_map = {code: index for index, code in index_code_map.items()}

# Get list of training and validation examples and labels
training_dir = paths.training
validation_dir = paths.validation
num_classes = len(code_index_map)


def get_paths_and_classes(root_dir, code_index_map=code_index_map):
    classes = []
    example_paths = []

    class_dirs = [class_dir for class_dir in os.listdir(root_dir)
                  if class_dir[-4:] != ".tar"]

    for num, class_dir in enumerate(class_dirs):

        full_class_path = root_dir + class_dir + '/'
        class_images = [class_dir + '/' + image for image in os.listdir(full_class_path)
                        if image[-5:] == ".JPEG"]

        example_paths.extend(class_images)
        class_index_list = [code_index_map[class_dir]]
        classes.extend(len(class_images) * class_index_list)

        assert len(example_paths) == len(classes), "Number of labels must match number of examples"

    # https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    classes_array = np.array([classes]).reshape(-1)
    one_hot_classes = np.eye(num_classes)[classes_array]

    return example_paths, one_hot_classes


x_paths_train, y_labels_train = get_paths_and_classes(training_dir)
x_paths_val, y_labels_val = get_paths_and_classes(validation_dir)

train_indices = np.random.choice(np.arange(len(x_paths_train)), size=len(x_paths_train), replace=False)
val_indices = np.random.choice(np.arange(len(x_paths_val)), size=len(x_paths_val), replace=False)

# Shuffle parameter in fit generator only shuffles minibatch order. Without shuffling ahead
# of time mini-batches will tend to be extremely correlated (i.e. same class) which will make
# our gradient estimate biased.
x_paths_train, y_labels_train = np.array(x_paths_train)[train_indices].tolist(), y_labels_train[train_indices]
x_paths_val, y_labels_val = np.array(x_paths_val)[val_indices].tolist(), y_labels_val[val_indices]

train_gen = AlexNetSequence(x_paths_train,
                            y_labels_train,
                            128,
                            paths.training,
                            np.array(eigenvectors.tolist()),
                            np.array(eigenvalues.tolist()),
                            np.array(pixel_avg.tolist()),
                            np.array(stdev.tolist()),
                            'train')

val_gen = AlexNetSequence(x_paths_val,
                          y_labels_val,
                          128,
                          paths.validation,
                          np.array(eigenvectors.tolist()),
                          np.array(eigenvalues.tolist()),
                          np.array(pixel_avg.tolist()),
                          np.array(stdev.tolist()),
                          'validate')

test_gen = AlexNetSequence(x_paths_val,
                           y_labels_val,
                           13,  # Actual batch size is 10x because of TTAs
                           paths.validation,
                           np.array(eigenvectors.tolist()),
                           np.array(eigenvalues.tolist()),
                           np.array(pixel_avg.tolist()),
                           np.array(stdev.tolist()),
                           'test')