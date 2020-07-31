from keras.utils import Sequence
import numpy as np
import copy
import shared.agent_methods.methods as agmeth
from abc import ABCMeta, abstractmethod


class SynchronousSequence(Sequence, metaclass=ABCMeta):
    def __init__(self, policy_source, source_type, environment, graph,
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
        self.graph = graph
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

        if self.use_double_dqn:
            assert self.use_target_model, "`use_double_dqn` cannot be set to `True` if no target model used"

        # Buffers
        self.reward_buffer = []
        self.feature_buffer = []
        self.action_buffer = []
        self.done_buffer = []
        self.valid_buffer = []

        # Iteration state variables
        self.episode = 1
        self.iteration = 0
        self.mem_frame_counter = 0  # We use a separate counter to count the number of frames added to replay memory

        # Keep track of state before getting minibatch, Initialize state buffers.
        prev_observation, prev_action, prev_reward, prev_done = self.environment.reset(), None, None, None
        self.record(prev_observation, prev_reward, prev_done, prev_action)

        # Model copies
        self.current_model = policy_source
        self.target_model = copy.deepcopy(self.current_model) if self.use_target_model else self.current_model

        assert source_type in ['value', 'policy'], "Allowed policy source types are `value` and `policy`"

        # Initial simulations
        for sim in range(self.replay_buffer_min):
            self.simulate_single()

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
        intra_epoch_iterations = ((iter - self.replay_buffer_min) / self.grad_update_frequency - 1) % self.epoch_length
        assert intra_epoch_iterations == idx, \
            "Consistency check, iterations ({}) and minibatch index ({}) don't match".format(intra_epoch_iterations, idx)
        return self.get_minibatch()

    @abstractmethod
    def observation_preprocess(self, observation):
        """
        Preprocesses observation into feature before it gets
        recorded in the replay memory. The feature output by
        this method is what gets saved to the replay memory.

        We found that doing part of the feature processing
        before saving features to the replay buffer was
        significantly more memory and compute efficient.
        Hence why we do some of the feature generation here
        and some in the get_feature_by_index method.
        """
        pass

    @abstractmethod
    def get_feature_at_index(self, index):
        """
        Gets the feature at a particular index. Post-processing
        of features is handled by this method. Note some feature
        processing is done by the obseravtion_preprocess method.
        See the documentation for that method for more information.
        """
        pass

    @abstractmethod
    def get_reward_at_index(self, index):
        """
        Gets the reward (transformed or not) at index
        """
        pass

    def get_done_at_index(self, index):
        return self.done_buffer[index]

    def get_action_at_index(self, index):
        return self.action_buffer[index]

    def get_latest_feature(self):
        """
        Gets the latest feature
        """
        max_index = len(self.feature_buffer) - 1
        return self.get_feature_at_index(max_index)

    def get_latest_reward(self):
        """
        Gets latest reward
        """
        max_index = len(self.feature_buffer) - 1
        return self.get_reward_at_index(max_index)

    def get_latest_done(self):
        """
        Get latest done indicator
        """
        max_index = len(self.feature_buffer) - 1
        return self.get_done_at_index(max_index)

    def get_latest_action(self):
        """
        Returns the latest *past* action. This differs from
        the `get_action` method which gets the *next* action.
        """
        max_index = len(self.feature_buffer) - 1
        return self.get_action_at_index(max_index)

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
        is_initial_buffer_fill_done = self.iteration >= self.replay_buffer_min
        is_stack_crossing_episodes = any([done is None for done in self.done_buffer[-self.get_states_start():]])

        # Do random policy until we have sufficiently filled the replay buffer
        # or when the latest frame stack is crossing between episodes
        if not is_initial_buffer_fill_done or is_stack_crossing_episodes:
            action = self.environment.action_space.sample()
        else:
            states = [self.get_latest_feature()]
            action = agmeth.get_action(self.current_model, self.environment, states,
                                       self.epsilon, self.iteration, self.graph)

        return action

    def update_target(self):
        """
        Update target model. Note we make sure
        this new model is created and ran on the
        same graph as the model we copy it from.
        """
        with self.graph.as_default():
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

    def record_valid(self):
        # Check validity by looking at current state plus previous self.get_state_start - 1 states
        min_index = max(-len(self.done_buffer), -self.get_states_start())  # At beginning buffer is shorter
        is_invalid = any([self.done_buffer[index] is None for index in range(min_index, 0)])

        if not is_invalid:
            # If the index is valid add it to the valid index queue
            self.valid_buffer.append(self.mem_frame_counter)
        if len(self.valid_buffer) > 0 \
                and self.mem_frame_counter - self.valid_buffer[0] >= self.replay_buffer_size - self.get_states_start():
            # If the index in the far left of the queue is too old, pop it. Note this only
            # works because this if check corresponds to a single increment of mem_frame_counter.
            # We subtract self.get_states_start() to account for the frame stack length.
            self.valid_buffer.pop(0)

    def record(self, observation, reward, done, action):
        """
        Takes in observed states and rewards and handles
        storing them in a buffer (which may be an array,
        a list of files to be loaded, etc.)

        :return:
        """
        # FIXME - for large enough buffers should we overwrite this method with one that serializes states???
        feature = self.observation_preprocess(observation)
        SynchronousSequence.record_single(self.feature_buffer, feature, self.replay_buffer_size)   # This is s_{t}
        SynchronousSequence.record_single(self.reward_buffer, reward, self.replay_buffer_size)  # This is r_{t-1}
        SynchronousSequence.record_single(self.action_buffer, action, self.replay_buffer_size)  # This is a_{t-1}
        SynchronousSequence.record_single(self.done_buffer, done, self.replay_buffer_size)  # whether s_{t} is terminal

        # Update buffer of valid memory indices (measured in frame counts stored in memory)
        self.record_valid()
        self.mem_frame_counter += 1

    def simulate_single(self):
        """
        Do a single iteration of simulation of the environment.

        :return: The next observation, action, reward, done state
        """
        # List indexing is O(1) so we're not concern about performance hits of getting last state from buffer
        prev_done = self.get_latest_done()
        prev_action = self.get_latest_action()

        self.iteration += 1
        repeat_action = (self.iteration % self.action_repeat) != 0

        # Update target after the appropiate number of iterations
        if self.use_target_model and (self.iteration % self.target_update_frequency) == 0 \
                and self.iteration > self.replay_buffer_min:
            self.update_target()

        # Check if episode done, if so draw next state with reset method on env
        if prev_done:
            observation, action, reward, done = self.environment.reset(), None, None, None
            self.episode += 1
        # Otherwise only get action every action_repeat iterations or on restart and use action to get next state
        else:
            # Get a new action after repeating action_repeat # times
            if not repeat_action or prev_action is None:
                # The previous action getter uses the previous observation which we get from the replay buffer
                action = self.get_action()
            else:
                action = prev_action

            observation, reward, done, info = self.environment.step(action)

        # The record method takes care of recording observed states
        self.record(observation, reward, done, action)

    def simulate(self):
        """
        Simulate grad_update_frequency steps of the agent
        and add them to the experience buffer.

        This method handles keeping track of iterations,
        updating the target model when appropriate, updating
        actions when required, etc.

        :return:
        """

        # Simulate grad_update_frequency # of environment and action steps
        for iter in range(self.grad_update_frequency):
            # The simulate single method carries out a single simulation step
            self.simulate_single()

        # Return iteration number for consistency assertion in batch getter
        return self.iteration

    @abstractmethod
    def get_states_start(self):
        pass

    def get_states_length(self):
        return min(len(self.feature_buffer), self.replay_buffer_size)

    def check_is_end_start_transition(self, index):
        """
        Check indices does not map to an transition from
        final state to initial. These are not allowed.

        Note that if get_states_start returns 1 then
        this will be a length 1 list with a single state,
        reducing to the classical case.
        """
        actions = self.action_buffer[index - self.get_states_start() + 1: index + 1]
        rewards = self.action_buffer[index - self.get_states_start() + 1: index + 1]
        check_list = []

        for action, reward in zip(actions, rewards):

            both_not_none = action is not None and reward is not None
            both_none = action is None and reward is None
            assert both_none or both_not_none, "Consistency check, reward and action must both be `None` or not."
            check_list.append(both_none)

        # We check that any of the states in the range searched define an episode end
        return any(check_list)

    def sample_indices(self):
        # Sample indices of the valid_buffer list itself
        valid_buffer_length = len(self.valid_buffer)
        sampled_buffer_indices = np.random.randint(valid_buffer_length, size=self.batch_size)

        # Use valid_buffer indices to sample actual valid index values.
        # Since lists have O(1) indexing  this approach is efficient.
        buffer_length = self.get_states_length()
        sampled_memory_indices = [buffer_length + (self.valid_buffer[index] - self.mem_frame_counter)
                                  for index in sampled_buffer_indices]

        return sampled_memory_indices

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
        rewards = np.array([self.get_reward_at_index(index) for index in sampled_indices])
        is_next_terminals = np.array([self.done_buffer[index] for index in sampled_indices])

        Q_max = agmeth.evaluate_state(self.target_model, next_states, self.graph, self.double_dqn_model())

        # We reshape the arrays so that it is clear to Tensorflow that each row is a datapoint
        x = [states, actions.reshape(-1, 1)]
        y = (rewards + self.gamma * Q_max * np.logical_not(is_next_terminals)).reshape(-1, 1)

        return x, y
