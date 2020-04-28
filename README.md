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
- Search files for filepaths, update to match evolving project structure. **X**
- Define which files will be shared and which will be model specific **X**
- Create project structure where shared files live in a `shared` folder
  and model specific files (including notes) live in model specific folders. **X**
- Define necessary abstractions for shared files, and define implementation 
  of abstractions (shared classes, child classes, etc.). **X**
- Define framework for configuration based model definition. Look up what
  is already built into Keras and Tensorflow. **X**
- Implement reorganizations. (Medium task) **X**
- Implement abstractions (Large task).
    - Parametrize runners
    - Parametrize sequence normalization and augmentations
    - Parametrize model run parameters like batch size, epochs, etc.
- Implement the next models in the roadmap. Make sure that we
  are able to still run and train Alexnet correctly (Large task).
  
### Notes on shared classes

**TL;DR**: We do not have to change the image preprocessing. The image import 
  and generation can stay largely the same. All of the models subtract the mean
  and take a random 224x224 crop from the preprocessed image. The augmentation
  may vary across models, and such maybe we want to abstract out the augmentations
  and parametrize the normalization so it can be toggled. 

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

We probably want to change the train and test runners from being scripts to being
callable functions with parameters. This way we can use the parameters to specify
the model file, augmentations, and train parameters we want to run with.

Not sure yet what to do about augmentations, normalization, or training settings.
We likely can just make augmentations into functions that we pass to the runner. 
We can make normalization into something that can be toggled on and off in the
generator class. As for the training settings maybe we can do these as a config 
file since there's only a handful to specify, and we can pass this configuration
to the runner method.

#### Models


- Keras does have built in methods for writing models out as configuration (`model.get_config()`).
- However the printout is extremely verbose, so much so it's harder to read than the code itself.
- Given this we think it's actually better to create a new copy of the model source code
  every time we update the model. Then we can store the model code, the checkpoints, and our
  notes in a single folder corresponding to that run for the model.
- We can use the terminal command `vimdiff` to get the diff between model files. This command 
  works very well.
- May be good to write a shell script for cleaning up checkpoints so we only keep the last one. 
  Otherwise we'll eat up memory unnecessarily. 

#### Augmentations, Normalization, & Training Settings

- How do we keep track of changes in these?
- Maybe the augmentation function can live with the model, and we can pass it to the generator
  through the runnner?

#### Runners

- Likely want to parametrize how model gets passed to runner.
- Also likely want to parametrize sequence generation so we can change up the 
  function used to generate the augmentations.


