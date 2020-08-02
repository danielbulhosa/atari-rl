from shared.generators.synchronous_sequence import SynchronousSequence
import copy


class EnvironmentSequence(SynchronousSequence):

    def get_states_start(self):
        return 1

    def observation_preprocess(self, observation):
        return observation

    def get_feature_at_index(self, index):
        return self.feature_buffer[index]

    def get_reward_at_index(self, index):
        return self.reward_buffer[index]

    def create_validation_instance(self, exploration_schedule):
        """
        Constructs a new sequence class for simulations
        done for evaluation. We pass a constructor to
        the evaluation callback because it allows us to
        reset the sequence (by creating a new one) when
        doing a new evaluation.
        """

        return EnvironmentSequence(self.current_model,
                                   source_type='value',
                                   environment=copy.deepcopy(self.environment_copy),
                                   graph=self.graph,
                                   epsilon=exploration_schedule,
                                   batch_size=0,
                                   grad_update_frequency=0,
                                   target_update_frequency=None,
                                   gamma=self.gamma,
                                   epoch_length=0,
                                   replay_buffer_size=1,
                                   replay_buffer_min=0)
