from keras.utils import Sequence
import numpy as np
import albumentations
import cv2
import math


class EnvironmentSequence(Sequence):
    def __init__(self, policy_source, environment, batch_size, source_type):
        """
        We initialize the EnvironmentSequence class
        with a batch size and an environment to generate
        data from. The generator will have a pointer to
        the model object in order to generate policy
        outputs.
        """
        # FIXME - is the circularity of passing model to generator which gets passed to model itself a problem?

        self.policy_source = policy_source
        self.environment = environment
        self.batch_size = batch_size
        self.source_type = source_type

        assert source_type in ['value', 'policy'], "Allowed policy source types are `value` and `policy`"

    def __len__(self):
        """
        # FIXME - complete, see docstring
        Should return the number of minibatches.
        But how do we define this appropiately in
        the RL case?

        This method is required by the Sequence
        interface.
        """

        raise NotImplementedError  # return number_of_minibatches

    def __getitem__(self, idx):
        """
        # FIXME - complete, see docstring
        The second method required to be implemented by
        the Sequence interface. This is the method used to
        generate minibatches.

        This should use the environment to generate more
        observations. This would also be a good place to
        safe observations to a replay memory and to
        multi-thread asynchronous agents.
        """

        raise NotImplementedError  # return x_minibatch_array, y_minibatch_array
