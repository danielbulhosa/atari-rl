# ILSVRC Canonical Model Implementation

This repo started as a project to learn Tensorflow by implementing Alexnet. 
Our eventual goal was to create other repos for other canonical models we
wanted to implement. As we learned more about these models we realized that
they leveraged the same dataset. Thus a lot of the same infrastructure and
code could be shared between the models. We are interested in implementing:

- GoogLeNet
- ResNet
- Others?

Our plan is to restructure this code to abstract away parts of it that can be
reused across models. In particular these parts should be reusable:

- Preprocessors
- Generators
- Training & Testing Runners

We also plan to use this as an opportunity to improve how we run experiments.
Currently we manually record experiments and the corresponding parameters by
writing them down in logs. This can become repetitive and (more importantly)
imprecise. We would like to specify model parameters and settings in separate
configuration files that we can commit and diff with each other. This should
allow for better experiment tracking.

These are the next steps to make this happen:
- Implement folder structure within model folders to keep track
  of versions.
- Implement the next models in the roadmap. Make sure that we
  are able to still run and train Alexnet correctly (Large task).
  
Optional:
- Parametrize normalization
- Parametrize subset of classes used for training
- Write a shell script for cleaning up checkpoints so we only keep the last one. 
  Otherwise we'll eat up memory unnecessarily. 
  
### Notes on new models (augmentation, settings, etc.)

- Need to look more closely, but in the case of GoogLeNet it seems the 
  preprocessing is the same as for Alexnet. The input is 224x224, cropped
  from the larger image from Imagenet like in the Alexnet model. They do
  not exactly specify image sampling but refer to other authors. We can 
  play with this on our own do and do some tuning.
- The preprocessing in ResNet is very close to that of AlexNet. The input
  image is 224x224 and cropped out of a preprocessed images. Details of
  augmentation may vary but augmentations are easier to abstract out later.
- If the processing is the same for both it significantly reduces the
  amount of refactoring we have to do for the generation and augmentation
  pipelines.
  
### Notes on tracking model changes:

**TL;DR**: Within master model folder, have one folder per version of the model run.
The folder contains a copy of the Python model file, notes about that version, 
and a folder with checkpoints (or just the last checkpoint).

- Will implement this as part of PR implementing new models. This way we can
  experiment more before reformatting all of the Alexnet work.

#### Notes

- We can use the terminal command `vimdiff` to get the diff between model files. This command 
  highlights changes intuitively.