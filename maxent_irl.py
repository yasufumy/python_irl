import numpy as np


class Reward:
    def __init__(self, n_features):
        self.n_features = n_features
        self.theta = np.random.uniform(size=(n_features,))

    def __call__(self, feature_matrix):
        if type(feature_matrix) is int:
            feature_matrix = np.eye(self.n_features, 1, feature_matrix).T
        return feature_matrix.dot(self.theta)

    def update(self, grad):
        self.theta += grad


class StateVisitationFrequency:
    def __init__(self, n_states, n_actions, probs):
        self.n_states = n_states
        self.n_actions = n_actions
        self.probs = probs

    def __call__(self, policy, trajectories):
        n_states = self.n_states
        n_actions = self.n_actions
        probs = self.probs
        n_trajectories, n_steps = trajectories.shape

        mu = np.zeros((n_steps, n_states))
        for trajectory in trajectories:
            mu[0, trajectory[0]] += 1
        mu /= n_trajectories

        for t in range(1, n_steps):
            for action in range(n_actions):
                for state in range(n_states):
                    for prob, next_state, _, _ in probs[state][action]:
                        mu[t][next_state] += mu[t-1][state] * policy[state][action] * prob

        return mu.sum(axis=0)


def compute_experts_feature(n_features, trajectories):
    n_trajectories, n_steps = trajectories.shape

    def one_hot_encoder(array):
        ncols = n_features
        out = np.zeros((array.size, ncols))
        out[np.arange(array.size), array.ravel()] = 1
        out.shape = array.shape + (ncols,)
        return out

    one_hot_trajectories = one_hot_encoder(trajectories)

    return one_hot_trajectories.sum(axis=(0, 1)) / n_trajectories
