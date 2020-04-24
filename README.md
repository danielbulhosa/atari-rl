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
- Search files for filepaths, update to match evolving project structure.
- Define which files will be shared and which will be model specific
- Create project structure where shared files live in a `shared` folder
  and model specific files (including notes) live in model specific folders.
- Define necessary abstractions for shared files, and define implementation 
  of abstractions (shared classes, child classes, etc.).
- Implement the next model in the roadmap, GoogLeNet. Make sure that we
  are able to still run and train Alexnet correctly.