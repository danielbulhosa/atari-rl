from keras.utils import Sequence
import numpy as np
import copy


class EnvironmentSequence(Sequence):
    def __init__(self, policy_source, source_type, environment, batch_size,
                 grad_update_frequency, target_update_frequency, action_repeat,
                 gamma, epoch_length, replay_buffer_size=None):
        """
        We initialize the EnvironmentSequence class
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
        # FIXME - is the circularity of passing model to generator which gets passed to model itself a problem?

        self.policy_source = policy_source
        self.source_type = source_type
        self.environment = environment
        self.batch_size = batch_size
        self.grad_update_frequency = grad_update_frequency
        self.target_update_frequency = target_update_frequency
        self.action_repeat = action_repeat
        self.gamma = gamma
        self.epoch_length = epoch_length
        self.replay_buffer_size = replay_buffer_size if replay_buffer_size is not None else batch_size

        # Buffers
        self.reward_buffer = []
        self.observation_buffer = []
        self.action_buffer = []
        self.done_buffer = []

        # Iteration state variables
        self.episode = 1
        self.iteration = 0

        # Keep track of state before getting minibatch, Initialize state buffers.
        self.prev_observation, self.prev_action, self.prev_reward, self.prev_done = self.environment.reset(), None, None, False
        EnvironmentSequence.record_single(self.observation_buffer, self.prev_observation, self.replay_buffer_size + 1)
        EnvironmentSequence.record_single(self.action_buffer, self.prev_action, self.replay_buffer_size)
        EnvironmentSequence.record_single(self.reward_buffer, self.prev_reward, self.replay_buffer_size + 1)
        EnvironmentSequence.record_single(self.done_buffer, self.prev_done, self.replay_buffer_size + 1)

        # Model copies
        self.current_model = policy_source
        self.target_model = copy.deepcopy(self.current_model)  # FIXME: Not sure this will work at all...

        assert source_type in ['value', 'policy'], "Allowed policy source types are `value` and `policy`"

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
        assert (iter / self.grad_update_frequency - 1) == idx, "Consistency check, iterations and minibatch index don't match"
        return self.get_minibatch()

    def get_latest_observations(self, n):
        """
        Gets the latest n observations and
        rewards.
        """
        assert 0 < n <= self.replay_buffer_size, "Cannot get more observations than stored in buffer"

        return self.observation_buffer[-n:]

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
        # FIXME - This will need to be different when the states are not the same as the observations.
        # FIXME- also note we assume `policy_source` is a Q function. This will not work with policy gradients then.
        states = self.get_latest_observations(1)
        actions = [0 for state in states]  # We don't actually care about indexing Q_0

        # FIXME - list generated here is empty. Figure is something wrong with simulate method.
        Q_a = self.current_model.predict([np.array(states), np.array(actions)])[0]
        max_actions = np.argwhere(Q_a == np.amax(Q_a)).flatten()
        action = np.random.choice(max_actions)

        return action

    def update_target(self):
        """
        Update target model.
        """

        self.target_model = copy.deepcopy(self.current_model)  # FIXME: Not sure this will work at all...

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
        EnvironmentSequence.record_single(self.observation_buffer, observation, self.replay_buffer_size + 1)
        # This is r_{t-1}
        EnvironmentSequence.record_single(self.reward_buffer, reward, self.replay_buffer_size)
        # This is a_{t-1}
        EnvironmentSequence.record_single(self.action_buffer, action, self.replay_buffer_size)
        # This denotes whether s_{t} is terminal
        EnvironmentSequence.record_single(self.done_buffer, done, self.replay_buffer_size + 1)

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
            if (self.iteration % self.target_update_frequency) == 0:
                self.update_target()

            # Check if episode done, if so draw next state with reset method on env
            if done:
                observation, action, reward, done = self.environment.reset(), None, None, False
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

    def get_states_length(self):
        return min(len(self.observation_buffer), self.replay_buffer_size)

    def sample_indices(self):

        return np.random.choice(np.arange(1, self.get_states_length()), self.batch_size)

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
        # FIXME - need to replace with method that feeds features, NOT observations to model
        # FIXME- also note we assume `policy_source` is a Q function. This will not work with policy gradients then.

        # We assume the paper does sampling with replacement. Makes the most sense if we're sampling a distribution.
        sampled_indices = self.sample_indices()

        states = np.array([self.observation_buffer[index - 1] for index in sampled_indices])
        actions = np.array([self.action_buffer[index] for index in sampled_indices])
        next_states = np.array([self.observation_buffer[index] for index in sampled_indices])
        rewards = np.array([self.reward_buffer[index] if self.reward_buffer[index] is not None else 0
                            for index in sampled_indices])
        is_next_terminals = np.array([self.done_buffer[index] for index in sampled_indices])

        dummy_actions = np.array([0 for state in next_states])  # Needed because of structure of model
        all_Q = self.target_model.predict([next_states, dummy_actions])
        Q_max = np.max(all_Q[0], axis=1)

        x = [states, actions]
        y = rewards + self.gamma * Q_max * np.logical_not(is_next_terminals)

        return x, y
