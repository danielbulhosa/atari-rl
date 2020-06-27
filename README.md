# Atari Deep RL Implementations

The purpose of this repository is to implement the Atari
video game environment using Open AI's gym library and
feeding environment outputs to a Tensorflow model to
create DQN and A3C learning systems.

The Open AI gym offers a uniform interface for passing
actions to an environment and getting states and rewards
in return. We believe we can combine this with Keras 
custom generators in order to build a complete deep learning
RL system. 

We're purposely not looking at implementations
online to figure this out on our own, otherwise we'd end
up imitating a lot of what's been done. We took a quick 
glance at [keras-rl](https://github.com/keras-rl/keras-rl)
to do at least a little bit of due diligence. It looks
like the author's approach was to create classes corresponding
to each RL model type, with the class handling things like
replay memory, forward and back propagation, storing the
actual Keras model, etc. This approach in essence uses
custom model classes to supercede the actual Keras
model and environment.

Instead of going all the way and replicating what keras-rl
did, we will supercede the environment with a custom generator,
but keep the Keras model separate of any custom classes. This
should be simpler to design and complete, for our learning
purposes.

We envision the following structure for our project:

+ An abstract generator class defining at a high level
  how generators pulling data from environments should
  work.
+ High level mixers that extend the functionality of the
  abstract generator by allowing multi-threading for
  example, for asynchronous training. Mixers may also
  be used to add a replay memory for example.
+ A child generator specifying the actual environment
  and generation scheme for a particular algorithmic
  implementation.
+ A simple model file specifying the model architecture
  used for training an agent. Models used for  value or 
  policy calculation tend to be a lot simpler than DL
  models used in other applications. As such we don't
  believe we'll need a lot of complexity for this.
  (We may borrow some of what we did for ILSVRC model
   configuration).
   
If this approach works for Atari and the framework is
applicable to other RL projects like simulated robotics
or self-driving we'll likely extend the repo to address
those use cases.

We expect the replay memory to be easy to implement, 
although it may push our hardware to a limit. Conversely
we think our hardware will comfortably handle asynchronous
training, but implementing this as a generator may be
tricky. Hopefully this all works!