import numpy as np


def evaluate_state(model, states):
    """
    Uses the `model` passed to function to calculate
    max_a Q(s, a) where s is equal to the `states`
    passed to the function.
    """

    dummy_actions = np.array([0 for state in states])  # Needed because of structure of model
    all_Q = model.predict([states, dummy_actions])
    Q_max = np.max(all_Q[0], axis=1)

    return Q_max


def get_action(model, environment, state, epsilon, iteration):
    """
    Uses `model` estimating the Q function to select
    the next optimal action based on the current `state`.

    :param model: A Keras model estimating the Q function.
    :param environment: The Open AI gym class representing the environment.
    :param state: The current state of the environment.
    :param epsilon: A function taking in an iteration and returning the
           current value of epsilon (i.e. the probability of choosing an
           action randomly to explore).
    :param iteration: The current iteration at which the simulations are in.
           Used for scheduling epsilon,
    :return: The next action taken by the agent.
    """

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

