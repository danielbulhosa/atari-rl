import keras.callbacks as call
import numpy as np
import tensorflow as tf
import shared.agent_methods.methods as agmeth


class EvaluateAgentCallback(call.Callback):

    def __init__(self, environment, gamma, tb_callback,
                 epsilon=(lambda iteration: 0.05), num_episodes=30,
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

        :param environment: The environment the agent (model) is
               being trained on.
        :param gamma: The discount rate for rewards. Used for
               calculating total episode rewards.
        :param tb_callback: The Tensorboard callback. We use this
               callback to write our evaluation metrics to the
               Tensorboard.
        :param epsilon: A function taking in the current iteration
               and returning the epsilon used to choose between
               exploration and exploitation.
        :param num_episodes: The number of episodes to evaluate the
               agent on.
        :param num_init_samples: The number of initial states on which
               to sample the policy value on.
        """

        self.environment = environment
        self.gamma = gamma
        self.tb_callback = tb_callback
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.num_init_samples = num_init_samples
        self.num_max_iter = num_max_iter
        self.step_number = 0

        assert num_episodes is not np.inf or num_max_iter is not np.inf, \
            "`num_episodes` and `num_max_iter` cannot both be equal to `np.inf`"

    def simulate_episodes(self, action_getter):
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
        episode = 0

        while episode < self.num_episodes:
            iteration = 0
            discount_factor = 1
            total_reward = 0
            observation = self.environment.reset()
            done = False

            while not done and iteration <= self.num_max_iter:
                iteration += 1
                action = action_getter(observation, iteration)
                observation, reward, done, info = self.environment.step(action)
                total_reward += discount_factor * reward
                discount_factor *= self.gamma

            # If episode was not completed and max iterations reached then do not count the episode and break
            if not done and iteration > self.num_max_iter:
                break
            # Otherwise count the episode and see if we reached the max iterations yet, loop or exit accordingly
            else:
                total_rewards.append(total_reward)
                episode_lenghts.append(iteration)

                if iteration > self.num_max_iter:
                    break

            episode += 1

        average_reward = np.mean(np.array(total_rewards))
        average_episode_length = np.mean(np.array(episode_lenghts))
        num_episodes = episode

        return average_reward, average_episode_length, num_episodes

    def on_train_begin(self, logs=None):
        average_reward, average_episode_length, average_num_episodes = self.simulate_episodes(self.get_action)
        random_reward, random_episode_length, random_num_episodes = self.simulate_episodes(self.get_random_action)

        print("\nInitial valuation. " +
              "\nAverage reward of initial policy over {} episodes: {}".format(self.num_episodes, average_reward) +
              "\nAverage reward of random policy over {} episodes: {}".format(self.num_episodes, random_reward) +
              "\nAverage episode length of initial policy over {} episodes: {}".format(self.num_episodes, average_episode_length) +
              "\nAverage episode length of random policy over {} episodes: {}".format(self.num_episodes, random_episode_length) +
              "\nNumber of episodes for initial policy over {} max iterations".format(average_num_episodes, self.num_max_iter) +
              "\nNumber of episodes for random policy over {} episodes: {}".format(random_num_episodes, self.num_max_iter))

    def on_epoch_end(self, epoch, logs=None):

        average_reward, average_episode_length, num_episodes = self.simulate_episodes(self.get_action)
        init_states = []

        for num in range(self.num_init_samples):
            observation = self.environment.reset()
            init_states.append(observation)

        init_sample_values = self.evaluate_state(np.array(init_states))
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
              "\nNumber of episodes over {} max iterations".format(num_episodes, self.num_max_iter) +
              "\nAverage expected value across {} initial states: {}".format(self.num_init_samples, average_init_value))

    def get_action(self, state, iteration):
        return agmeth.get_action(self.model, self.environment, [state], self.epsilon, iteration)

    def get_random_action(self, state, iteration):
        return self.environment.action_space.sample()

    def evaluate_state(self, states):
        return agmeth.evaluate_state(self.model, states)
