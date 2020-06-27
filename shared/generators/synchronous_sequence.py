from keras.utils import Sequence
import numpy as np
import copy
import shared.agent_methods.methods as agmeth
from abc import ABCMeta, abstractmethod


class SynchronousSequence(Sequence, metaclass=ABCMeta):
    def __init__(self, policy_source, source_type, environment,
                 epsilon, batch_size, grad_update_frequency,
                 target_update_frequency, action_repeat,
                 gamma, epoch_length, replay_buffer_size=None,
                 replay_buffer_min=None, use_double_dqn=False):
        """
        We initialize the SynchronousSequence class
        with a batch size and an environment to generate
        data from. The generator will have a pointer to
        the model object in order to generate policy
        outputs.

        Note that:

        - If replay_buffer_size > batch_size we effectively get an experience replay mechanism.
          If replay_buffer_size < batch_size we will resample points for the minibatch update.
          If replay_buffer_size == batch_size we sample the entire buffer.

        - If batch_size == grad_update_frequency each learning update uses entirely new data.
          If batch_size < grad_update_frequency (and replay_buffer_size == batch_size) then
          some experiences will be skipped and not contribute to the gradient calculation.
          If batch_size > grad_update_frequency more recent experiences get resampled.
        """

        self.policy_source = policy_source
        self.source_type = source_type
        self.environment = environment
        self.epsilon = epsilon  # This should be a function taking in the iteration number
        self.batch_size = batch_size
        self.grad_update_frequency = grad_update_frequency
        self.target_update_frequency = target_update_frequency
        self.action_repeat = action_repeat
        self.gamma = gamma
        self.epoch_length = epoch_length
        self.replay_buffer_size = replay_buffer_size if replay_buffer_size is not None else batch_size
        self.replay_buffer_min = replay_buffer_min if replay_buffer_min is not None else batch_size
        self.use_double_dqn = use_double_dqn
        self.use_target_model = self.target_update_frequency is not None
        self.initial_sims = self.replay_buffer_min // self.grad_update_frequency
        self.initial_iterations = self.initial_sims * self.grad_update_frequency

        if self.use_double_dqn:
            assert self.use_target_model, "`use_double_dqn` cannot be set to `True` if no target model used"

        # Buffers
        self.reward_buffer = []
        self.observation_buffer = []
        self.action_buffer = []
        self.done_buffer = []

        # Iteration state variables
        self.episode = 1
        self.iteration = 0

        # Keep track of state before getting minibatch, Initialize state buffers.
        self.prev_observation, self.prev_action, self.prev_reward, self.prev_done = self.environment.reset(), None, None, None
        SynchronousSequence.record_single(self.observation_buffer, self.prev_observation, self.replay_buffer_size)
        SynchronousSequence.record_single(self.action_buffer, self.prev_action, self.replay_buffer_size)
        SynchronousSequence.record_single(self.reward_buffer, self.prev_reward, self.replay_buffer_size)
        SynchronousSequence.record_single(self.done_buffer, self.prev_done, self.replay_buffer_size)

        # Model copies
        self.current_model = policy_source
        self.target_model = copy.deepcopy(self.current_model) if self.use_target_model else self.current_model

        assert source_type in ['value', 'policy'], "Allowed policy source types are `value` and `policy`"

        # Initial simulations

        for sim in range(self.initial_sims):
            self.simulate()

    def __len__(self):
        """
        Should return the number of minibatches
        per epoch. In the case of deep RL this
        is a choice of convenience.

        This method is required by the Sequence
        interface.
        """

        return self.epoch_length

    def __getitem__(self, idx):
        """
        The second method required to be implemented by
        the Sequence interface. This is the method used to
        generate minibatches.

        This should use the environment to generate more
        observations. This would also be a good place to
        safe observations to a replay memory and to
        multi-thread asynchronous agents.
        """

        # FIXME - what happens if we have more than one worker loading minibatches? Do we have asynchrony issues?
        iter = self.simulate()
        assert ((iter - self.initial_iterations) / self.grad_update_frequency - 1) % self.epoch_length == idx, \
            "Consistency check, iterations and minibatch index don't match"
        return self.get_minibatch()

    @abstractmethod
    def get_feature_at_index(self, index):
        """
        Gets the feature at a particular index
        """
        pass

    def get_latest_feature(self):
        """
        Gets the latest feature
        """
        max_index = len(self.observation_buffer) - 1
        return self.get_feature_at_index(max_index)

    def get_action(self):
        """
        Gets next action either from value function or
        policy function based on whatever is passed to
        the model.

        We get the latest experiences from the observation
        buffer using get_latest_observations method. This
        allows us to stack previous states if necessary.

        :return: Next action.
        """
        # FIXME- Note agmeth.get_action assumes `policy_source` is a Q func. This will not work with policy grads

        # Do random policy until we have sufficiently filled the replay buffer
        if self.iteration // self.grad_update_frequency < self.initial_sims:
            action = self.environment.action_space.sample()

        else:
            states = self.get_latest_feature()
            action = agmeth.get_action(self.current_model, self.environment, states, self.epsilon, self.iteration)

        return action

    def update_target(self):
        """
        Update target model.
        """
        self.target_model = copy.deepcopy(self.current_model) if self.use_target_model else self.current_model

    def double_dqn_model(self):
        if self.use_double_dqn:
            return self.target_model
        else:
            return None

    @staticmethod
    def record_single(buffer, new_value, length_limit):
        if len(buffer) == length_limit:
            buffer.pop(0)

        buffer.append(new_value)

    def record(self, observation, reward, done, action):
        """
        Takes in observed states and rewards and handles
        storing them in a buffer (which may be an array,
        a list of files to be loaded, etc.)

        :return:
        """
        # FIXME - for large enough buffers should we overwrite this method with one that serializes states???
        # Note the observation we are saving is actually s_t. We keep an extra state so we can sample transitions.
        SynchronousSequence.record_single(self.observation_buffer, observation, self.replay_buffer_size)
        # This is r_{t-1}
        SynchronousSequence.record_single(self.reward_buffer, reward, self.replay_buffer_size)
        # This is a_{t-1}
        SynchronousSequence.record_single(self.action_buffer, action, self.replay_buffer_size)
        # This denotes whether s_{t} is terminal
        SynchronousSequence.record_single(self.done_buffer, done, self.replay_buffer_size)

    def simulate(self):
        """
        Simulate grad_update_frequency steps of the agent
        and add them to the experience buffer.

        This method handles keeping track of iterations,
        updating the target model when appropriate, updating
        actions when required, etc.

        :return:
        """

        observation = self.prev_observation
        done = self.prev_done
        action = self.prev_action

        # Simulate grad_update_frequency # of environment and action steps
        for iter in range(self.grad_update_frequency):
            self.iteration += 1

            # Update target after the appropiate number of iterations
            if self.use_target_model and (self.iteration % self.target_update_frequency) == 0\
                    and self.iteration > self.initial_iterations:

                self.update_target()

            # Check if episode done, if so draw next state with reset method on env
            if done:
                observation, action, reward, done = self.environment.reset(), None, None, None
                self.episode += 1
            # Otherwise only get action every action_repeat iterations or on restart and use action to get next state
            else:
                # Get a new action after repeating action_repeat # times
                if (self.iteration % self.action_repeat) == 0 or action is None:
                    action = self.get_action()

                observation, reward, done, info = self.environment.step(action)

            # The record method takes care of recording observed states
            self.record(observation, reward, done, action)

        self.prev_observation, self.prev_done, self.prev_action = observation, done, action

        return self.iteration

    @abstractmethod
    def get_states_start(self):
        pass

    def get_states_length(self):
        return min(len(self.observation_buffer), self.replay_buffer_size)

    def check_is_end_start_transition(self, index):
        """
        Check index does not map to an transition from
        final state to initial. These are not allowed.
        """

        action = self.action_buffer[index]
        reward = self.action_buffer[index]
        both_not_none = action is not None and reward is not None
        both_none = action is None and reward is None

        assert both_none or both_not_none, "Consistency check, reward and action must both be `None` or not."

        return both_none

    def sample_indices(self):

        # check_is_end_transition is a method from the parent class
        valid_indices = [index for index in range(self.get_states_start(), self.get_states_length())
                         if not self.check_is_end_start_transition(index)]

        sampled_indices = np.random.choice(valid_indices, self.batch_size)

        return sampled_indices

    def get_minibatch(self):
        """
        A method for retrieving experiences from our
        experience buffer. It serves as the interface
        between our methods and the experience buffer.

        When the experience buffer is larger than
        grad_update_frequency it serves as a memory
        replay mechanism.

        :return: An experience minibatch.
        """
        # FIXME- Note agmeth.evaluate_state assumes `policy_source` is a Q func. This will not work with policy grads

        # We assume the paper does sampling with replacement. Makes the most sense if we're sampling a distribution.
        sampled_indices = self.sample_indices()

        states = np.array([self.get_feature_at_index(index - 1) for index in sampled_indices])
        actions = np.array([self.action_buffer[index] for index in sampled_indices])
        next_states = np.array([self.get_feature_at_index(index) for index in sampled_indices])
        rewards = np.array([self.reward_buffer[index] for index in sampled_indices])
        is_next_terminals = np.array([self.done_buffer[index] for index in sampled_indices])

        Q_max = agmeth.evaluate_state(self.target_model, next_states, self.double_dqn_model())

        # We reshape the arrays so that it is clear to Tensorflow that each row is a datapoint
        x = [states, actions.reshape(-1, 1)]
        y = (rewards + self.gamma * Q_max * np.logical_not(is_next_terminals)).reshape(-1, 1)

        return x, y
