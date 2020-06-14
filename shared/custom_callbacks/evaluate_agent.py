import keras.callbacks as call
import numpy as np
import tensorflow as tf
import shared.agent_methods.methods as agmeth


class EvaluateAgentCallback(call.Callback):

    def __init__(self, environment, gamma, tb_callback,
                 epsilon=(lambda iteration: 0.05), num_episodes=30,
                 num_init_samples=1000):

        self.environment = environment
        self.gamma = gamma
        self.tb_callback = tb_callback
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.num_init_samples = num_init_samples
        self.step_number = 0

    def on_epoch_end(self, epoch, logs=None):

        total_rewards = np.empty(self.num_episodes)
        init_states = []

        for episode in range(self.num_episodes):
            iteration = 0
            discount_factor = 1
            total_reward = 0
            observation = self.environment.reset()
            done = False

            while not done:
                iteration += 1
                action = self.get_action(observation, iteration)
                observation, reward, done, info = self.environment.step(action)
                total_reward += discount_factor * reward
                discount_factor *= self.gamma

            total_rewards[episode] = total_reward

        average_reward = np.mean(total_rewards)

        for num in range(self.num_init_samples):
            observation = self.environment.reset()
            init_states.append(observation)

        init_sample_values = self.evaluate_state(np.array(init_states))
        average_init_value = np.mean(init_sample_values)

        # Code borrowed from: https://chadrick-kwag.net/how-to-manually-write-to-tensorboard-from-tf-keras-callback-useful-trick-when-writing-a-handful-of-validation-metrics-at-once/
        items_to_write = {
            "average_reward": average_reward,
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
              "\nAverage expected value across {} initial states: {}".format(self.num_init_samples, average_init_value))

    def get_action(self, state, iteration):
        return agmeth.get_action(self.model, self.environment, [state], self.epsilon, iteration)

    def evaluate_state(self, states):
        return agmeth.evaluate_state(self.model, states)