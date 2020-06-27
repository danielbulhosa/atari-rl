from shared.generators.synchronous_sequence import SynchronousSequence
import numpy as np
import cv2


class AtariSequence(SynchronousSequence):

    def __init__(self, policy_source, source_type, environment,
                 n_stack, stack_dims, epsilon, batch_size,
                 grad_update_frequency, target_update_frequency, action_repeat,
                 gamma, epoch_length, replay_buffer_size=None,
                 replay_buffer_min=None, use_double_dqn=False):

        # How many frames to stack to create features
        self.n_stack = n_stack
        self.stack_dims = stack_dims

        super().__init__(policy_source, source_type, environment,
                         epsilon, batch_size, grad_update_frequency,
                         target_update_frequency, action_repeat,
                         gamma, epoch_length, replay_buffer_size,
                         replay_buffer_min, use_double_dqn)

    def get_states_start(self):
        # Plus one because for each frame we take its average
        # with the previous one to eliminate flickering artifacts
        return self.n_stack + 1

    def observation_preprocess(self, observation):
        """
        Preprocess an image by extracting the Y
        luminace channel and resizing them based
        on the dims in the tuple `self.stack_dims`.
        """

        img_yuv = cv2.cvtColor(observation, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        preprocessed_image = cv2.resize(y, (self.stack_dims[0], self.stack_dims[1]))
        return preprocessed_image

    def get_feature_at_index(self, index):
        """
        Based on passed observation (timestep relative
        to buffer start), compute features passed to agent.

        We replicate the preprocessing (phi) described in
        the Atari paper. Namely we:

        - Take the frame indexed and the `n_stack + 1`
          preceeding ones.
        - We take contiguous pairwise maxes of images
          to end with a stack of `n_stack` images with
          flickering eliminated.
        - We preprocess images by extracting the Y
          luminace channel and resizing them,
        """

        observation_stack = self.observation_buffer[index - self.n_stack:index + 1]
        maxed_observations = [np.maximum(observation_stack[i], observation_stack[i+1])
                              for i in range(len(observation_stack) - 1)]
        preprocessed_images = [self.observation_preprocess(observation)
                               for observation in maxed_observations]

        return preprocessed_images
