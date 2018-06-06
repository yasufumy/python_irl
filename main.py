from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.envs.registration import register

from value_iteration import ValueIteration, sample_trajectories
from maxent_irl import Reward, StateVisitationFrequency, compute_experts_feature

register(id='EasyFrozenLakeEnv-v0',  entry_point='gym.envs.toy_text:FrozenLakeEnv',
         kwargs={'is_slippery': False})


def train(gamma, epsilon, n_samples, n_steps, n_epochs, learning_rate):
    env = gym.make('EasyFrozenLakeEnv-v0')
    value_iteration = ValueIteration(env.nS, env.nA, env.P)

    # preparing an expert
    V, policy = value_iteration(gamma, epsilon)
    trajectories = sample_trajectories(env, policy, n_steps, n_samples)
    experts_feature = compute_experts_feature(env.nS, trajectories)

    # training
    feature_matrix = np.eye(env.nS)
    reward_function = Reward(env.nS)
    svf = StateVisitationFrequency(env.nS, env.nA, env.P)

    for i in range(n_epochs):
        V, policy = value_iteration(gamma, epsilon, reward_function)
        P = svf(policy, trajectories)
        grad = experts_feature - feature_matrix.T.dot(P)
        reward_function.update(learning_rate * grad)

    return reward_function(feature_matrix)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--epsilon', default=1e-5, type=float)
    parser.add_argument('--n-samples', default=100, type=int)
    parser.add_argument('--n-steps', default=10, type=int)
    parser.add_argument('--n-epochs', default=20, type=int)
    parser.add_argument('--learning-rate', default=0.1, type=float)
    args = parser.parse_args()

    reward = train(args.gamma, args.epsilon, args.n_samples, args.n_steps,
                   args.n_epochs, args.learning_rate)
    plt.pcolor(reward.reshape(4, 4)[::-1, :])
    plt.title(f'samples: {args.n_samples}, steps: {args.n_steps}')
    plt.colorbar()
    plt.show()
