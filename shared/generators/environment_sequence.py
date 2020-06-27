from shared.generators.synchronous_sequence import SynchronousSequence
import shared.agent_methods.methods as agmeth
import numpy as np


class EnvironmentSequence(SynchronousSequence):

    def get_states_length(self):
        return min(len(self.observation_buffer), self.replay_buffer_size)

    def sample_indices(self):

        # check_is_end_transition is a method from the parent class
        valid_indices = [index for index in range(1, self.get_states_length())
                         if not self.check_is_end_start_transition(index)]

        sampled_indices = np.random.choice(valid_indices, self.batch_size)

        return sampled_indices

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

        # Do random policy until we have sufficiently filled the replay buffer
        if self.iteration // self.grad_update_frequency < self.initial_sims:
            action = self.environment.action_space.sample()

        else:
            states = self.get_latest_observations(1)
            action = agmeth.get_action(self.current_model, self.environment, states, self.epsilon, self.iteration)

        return action