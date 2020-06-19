import numpy as np


def evaluate_state(model, states):
    dummy_actions = np.array([0 for state in states])  # Needed because of structure of model
    all_Q = model.predict([states, dummy_actions])
    Q_max = np.max(all_Q[0], axis=1)

    return Q_max


# FIXME - replace equivalent code in environment and agent evaluation classes
def get_action(model, environment, state, epsilon, iteration):
    current_epsilon = epsilon(iteration)  # Note epsilon is a method

    is_greedy = np.random.binomial(1, 1 - current_epsilon)

    if is_greedy:
        actions = [0 for state in state]  # We don't actually care about indexing Q_0
        assert len(state) == 1, "Only one state should be passed to this method"

        # Note in practice
        Q_a = model.predict([np.array(state), np.array(actions)])[0]
        # Note we need to reshape because output is 2D array.
        # Each 2D Array entry is the 2D index of a max. Since
        # We only pass one state max_states should have length
        # between 1 and # of possible actions.
        max_actions = np.argwhere(Q_a == np.amax(Q_a, axis=1).reshape(-1, 1))[:, 1]

        action = np.random.choice(max_actions)

    else:
        action = environment.action_space.sample()

    return action

