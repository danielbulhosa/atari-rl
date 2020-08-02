from shared.generators.synchronous_sequence import SynchronousSequence
import numpy as np
import cv2
import copy


class AtariSequence(SynchronousSequence):

    def __init__(self, policy_source, source_type, environment, graph,
                 n_stack, stack_dims, pair_max, epsilon, batch_size,
                 grad_update_frequency, target_update_frequency, action_repeat,
                 gamma, epoch_length, replay_buffer_size=None,
                 replay_buffer_min=None, use_double_dqn=False):

        # How many frames to stack to create features
        self.pair_max = pair_max
        self.n_stack = n_stack
        self.stack_dims = stack_dims

        super().__init__(policy_source, source_type, environment, graph,
                         epsilon, batch_size, grad_update_frequency,
                         target_update_frequency, action_repeat,
                         gamma, epoch_length, replay_buffer_size,
                         replay_buffer_min, use_double_dqn)

    def get_states_start(self):
        # Plus one when pair_max is true because for each frame we take
        # its average with the previous one to eliminate flickering artifacts
        if self.pair_max:
            return self.n_stack + 1
        else:
            return self.n_stack

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

        The preprocessing is done when we save features
        to the replay memory. This turned out to be significantly
        more compute efficient. Note that in the paper these
        operations are done in the opposite order but since our
        scaling operation is linear the two feature processing
        steps commute so we were able to change their order.
        """
        observation_stack = self.feature_buffer[index - (self.get_states_start() - 1):index + 1]

        # If no pair max is taken the operation is just the identity
        if self.pair_max:
            final_observations = [np.maximum(observation_stack[i], observation_stack[i+1])
                                  for i in range(len(observation_stack) - 1)]
        else:
            final_observations = observation_stack

        return np.array(final_observations).reshape((self.stack_dims[0], self.stack_dims[1], self.n_stack))

    @staticmethod
    def reward_transform(reward):
        """
        In the Atari paper rewards are either 1, 0, or -1
        based on their sign and equality to zero. This function
        does that transform.

        :param reward: The actial reward.
        :return: The rescaled reward.
        """
        if reward > 0:
            return 1
        elif reward < 0:
            return -1
        else:
            return 0

    def get_reward_at_index(self, index):
        # Since ALE handles skipping frames we do NOT sum rewards. We only take the reward at the index
        # ALE seems to return the reward differential between frames (and we assume accounts for skips?)
        return AtariSequence.reward_transform(self.reward_buffer[index])

    def create_validation_instance(self, exploration_schedule):
        """
        Constructs a new sequence class for simulations
        done for evaluation. We pass a constructor to
        the evaluation callback because it allows us to
        reset the sequence (by creating a new one) when
        doing a new evaluation.
        """
        return AtariSequence(self.current_model,
                             source_type='value',
                             environment=copy.deepcopy(self.environment_copy),
                             graph=self.graph,
                             n_stack=self.n_stack,
                             stack_dims=self.stack_dims,
                             pair_max=self.pair_max,
                             # We checked and the underlying simulator does NOT take pairwise max
                             epsilon=exploration_schedule,
                             batch_size=0,
                             # For validation we do not create batches, hence we do not use this parameter
                             grad_update_frequency=0,
                             target_update_frequency=None,
                             action_repeat=1,  # No need for manual action repeat, Gym environment handles repeats
                             gamma=self.gamma,
                             epoch_length=0,
                             replay_buffer_size=self.get_states_start(),  # Add one frame if we take pair maxes
                             replay_buffer_min=0,
                             )
