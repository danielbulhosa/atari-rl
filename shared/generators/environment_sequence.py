from shared.generators.synchronous_sequence import SynchronousSequence


class EnvironmentSequence(SynchronousSequence):

    def get_states_start(self):
        return 1

    def get_feature_at_index(self, index):
        return self.observation_buffer[index]

    def get_reward_at_index(self, index):
        return self.reward_buffer[index]