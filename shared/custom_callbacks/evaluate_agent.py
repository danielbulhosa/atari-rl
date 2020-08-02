import keras.callbacks as call
import numpy as np
import tensorflow as tf
import shared.agent_methods.methods as agmeth


class EvaluateAgentCallback(call.Callback):

    def __init__(self, sequence_constructor, tb_callback, num_episodes=30,
                 num_init_samples=1000, num_max_iter=np.inf):

        """
        Create a callback that evaluates the performance of
        the agent being trained on the `environment` passed.

        The performance is evaluated in two ways:
          - Run agent for `num_episodes` episodes, average total
            rewards across episodes.
          - Sample `num_init_samples` initial states, calculate
            value of those states, take their average.

        The latter approach was used in the original Deep RL Atari
        paper by D. Silver et al. as an alternative to the former
        as it was found to be an estimator of model performance with
        less variance.

        :param sequence_constructor: Constructor for the sequence class
               used to simulate the agent and the environment.
        :param tb_callback: The Tensorboard callback. We use this
               callback to write our evaluation metrics to the
               Tensorboard.
        :param num_episodes: The number of episodes to evaluate the
               agent on.
        :param num_init_samples: The number of initial states on which
               to sample the policy value on.
        :param num_max_iter: The number of maximum iterations to simulate
               for the purpose of evaluation.
        """

        self.sequence_constructor = sequence_constructor
        self.tb_callback = tb_callback
        self.num_episodes = num_episodes
        self.num_init_samples = num_init_samples
        self.num_max_iter = num_max_iter
        self.step_number = 0

        assert num_episodes is not np.inf or num_max_iter is not np.inf, \
            "`num_episodes` and `num_max_iter` cannot both be equal to `np.inf`"

    def simulate_episodes(self, sequence):
        """
        Code for simulating an episode. Used in evaluation
        callback for simulating episodes with agent and
        random policy.

        :param action_getter: Method for getting actions
        from agent.
        :return: Average rewards and episode lengths.
        """

        total_rewards = []
        episode_lenghts = []

        while sequence.episode <= self.num_episodes:
            discount_factor = 1
            total_reward = 0
            done = sequence.get_latest_done()
            assert not done, "Initial done of episode should always be `None` or `False`"
            iteration_start = sequence.iteration

            while not done and sequence.iteration <= self.num_max_iter:
                sequence.simulate_single()
                reward = sequence.get_latest_reward()
                done = sequence.get_latest_done()
                total_reward += discount_factor * reward
                discount_factor *= sequence.gamma

            # If episode was not completed and max iterations reached then do not count the episode and break
            if not done and sequence.iteration > self.num_max_iter:
                break
            # Otherwise count the episode and see if we reached the max iterations yet, loop or exit accordingly
            else:
                total_rewards.append(total_reward)
                episode_lenghts.append(sequence.iteration - iteration_start)
                # Simulate one more step to restart environment
                sequence.simulate_single()

                if sequence.iteration > self.num_max_iter:
                    break

        average_reward = np.mean(np.array(total_rewards))
        average_episode_length = np.mean(np.array(episode_lenghts))
        num_episodes = sequence.episode

        return average_reward, average_episode_length, num_episodes

    def on_train_begin(self, logs=None):
        # Need a new sequence each time to reset instance state
        policy_sequence = self.sequence_constructor()
        average_reward, average_episode_length, average_num_episodes = self.simulate_episodes(policy_sequence)
        random_sequence = self.sequence_constructor(random=True)
        random_reward, random_episode_length, random_num_episodes = self.simulate_episodes(random_sequence)

        print("\nInitial valuation. " +
              "\nAverage reward of initial policy over {} episodes: {}".format(self.num_episodes, average_reward) +
              "\nAverage reward of random policy over {} episodes: {}".format(self.num_episodes, random_reward) +
              "\nAverage episode length of initial policy over {} episodes: {}".format(self.num_episodes, average_episode_length) +
              "\nAverage episode length of random policy over {} episodes: {}".format(self.num_episodes, random_episode_length) +
              "\nNumber of episodes for initial policy over {} max iterations: {}".format(self.num_max_iter, average_num_episodes) +
              "\nNumber of episodes for random policy over {} max iterations: {}".format(self.num_max_iter, random_num_episodes))

    def on_epoch_end(self, epoch, logs=None):
        policy_sequence = self.sequence_constructor()
        average_reward, average_episode_length, num_episodes = self.simulate_episodes(policy_sequence)
        init_states = []

        for num in range(self.num_init_samples):
            # Reset sequence class, generate initial observation stack
            policy_sequence = self.sequence_constructor()
            for iteration in range(policy_sequence.get_states_start()):
                policy_sequence.simulate_single()
            # Get latest (in this case the first) feature
            observation = policy_sequence.get_latest_feature()
            init_states.append(observation)

        init_sample_values = self.evaluate_state(np.array(init_states), policy_sequence.graph)
        average_init_value = np.mean(init_sample_values)

        # Code borrowed from: https://chadrick-kwag.net/how-to-manually-write-to-tensorboard-from-tf-keras-callback-useful-trick-when-writing-a-handful-of-validation-metrics-at-once/
        items_to_write = {
            "average_reward": average_reward,
            "average_episode_length": average_episode_length,
            "number_of_episodes": num_episodes,
            "average_initial_state_value": average_init_value
        }
        writer = self.tb_callback.writer
        for name, value in items_to_write.items():
            summary = tf.summary.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            writer.add_summary(summary, self.step_number)
            writer.flush()
        self.step_number += 1

        print("\nEvaluation complete. " +
              "\nAverage reward over {} episodes: {}".format(self.num_episodes, average_reward) +
              "\nAverage episode length over {} episodes: {}".format(self.num_episodes, average_episode_length) +
              "\nNumber of episodes over {} max iterations: {}".format(self.num_max_iter, num_episodes) +
              "\nAverage expected value across {} initial states: {}".format(self.num_init_samples, average_init_value))

    def evaluate_state(self, states, graph):
        return agmeth.evaluate_state(self.model, states, graph)
