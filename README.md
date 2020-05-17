# ILSVRC Canonical Model Implementation

This codebase started as a project to learn Tensorflow & Keras by implementing Alexnet. 
Our eventual goal was to create implementations of other canonical models we
wanted to implement. As we learned more about these models we realized that
they leveraged the same dataset. Thus a lot of the same infrastructure and
code could be shared between the models. We currently have implemented:

- AlexNet
- GoogLeNet
- ResNet

## Shared Generators, Runners, and Constants

The different models leverage the same custom generator, runner, and precomputation
(constants) classes and scripts. Custom layers required for each model such as LRN, 
inception modules, or residual units are also implemented in this repo. All of the
shared codebase can be found in the `shared` module.

## Models & Experiment Tracking

A major challenge with the original codebase was keeping track of model versions
during different experiments. To address this issue we defined a project structure 
such that:

- Each model has its own folder, within the `models` module.
- Within each model folder each experiment has its own folder, denoted by a version number.
- Each version folder contains the source code (`rundef.py`) and parameters for that experiment,
  as well as logs and checkpoints.
  
The model source code contains both the model definition, as well as other variables that
get changed often during experiments such as:

- Non-shared augmentations
- Weight and bias regularization
- Mini-batch size
- Learning rate schedule (callback) definition
- Number of epochs
- Checkpoint and logging settings (callbacks)

In the original codebase, these different pieces of experiments were scattered across
different files. This made it hard to track changes. With the new file structure, 
we can use the terminal command `vimdiff` between the `rundef.py` files in different
version folders to see the changes between experiments!

Finally for convenience we parametrized the number of classes and the number of
minibatches used for training. A common debugging strategy for neural networks
is overfitting on a single minibatch or training on a subset of the classes. The
parametrization makes this simple to do and track.

## Training & Validation Data

The data used for training these models is the data for the ILSVRC 2012 competition.
The `imagenet` folder contains detailed instructions for how to get and preprocess
this dataset. It also contains scripts for structuring the data in a way that the
model and shared scripts know how to navigate.

## Potential Future Updates

- Parametrize normalization.
- Write a shell script for cleaning up checkpoints so we only keep the last one. 
  Otherwise we'll eat up memory unnecessarily. 
- Definite a cleaner, high level interface for shared classes.
- Generalize ResNet model architecture.