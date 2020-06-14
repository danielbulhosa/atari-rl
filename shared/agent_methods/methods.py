import numpy as np


def evaluate_state(model, states):
    dummy_actions = np.array([0 for state in states])  # Needed because of structure of model
    all_Q = model.predict([states, dummy_actions])
    Q_max = np.max(all_Q[0], axis=1)

    return Q_max


# FIXME - replace equivalent code in environment and agent evaluation classes
def get_action(model, environment, states, epsilon, iteration):
    current_epsilon = epsilon(iteration)  # Note epsilon is a method

    is_greedy = np.random.binomial(1, 1 - current_epsilon)

    if is_greedy:
        actions = [0 for state in states]  # We don't actually care about indexing Q_0

        Q_a = model.predict([np.array(states), np.array(actions)])[0]
        max_actions = np.argwhere(Q_a == np.amax(Q_a)).flatten()
        action = np.random.choice(max_actions)

    else:
        action = environment.action_space.sample()

    return action