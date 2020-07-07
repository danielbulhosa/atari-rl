from shared.generators.synchronous_sequence import SynchronousSequence


class EnvironmentSequence(SynchronousSequence):

    def get_states_start(self):
        return 1

    def observation_preprocess(self, observation):
        return observation

    def get_feature_at_index(self, index):
        return self.feature_buffer[index]

    def get_reward_at_index(self, index):
        return self.reward_buffer[index]