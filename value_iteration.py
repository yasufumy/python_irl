import numpy as np

import gym
from gym.envs.registration import register
register(id='EasyFrozenLakeEnv-v0',  entry_point='gym.envs.toy_text:FrozenLakeEnv',
         kwargs={'is_slippery': False})


class ValueIteration:
    def __init__(self, n_states, n_actions):
        self.V = np.zeros(n_states)
        self.n_states = n_states
        self.n_actions = n_actions

    def run(self, env, gamma, epslion):
        def compute_expected_reward(state):
            A = np.zeros(self.n_actions)
            for action in range(self.n_actions):
                for prob, next_state, reward, done in env.P[state][action]:
                    A[action] += prob * (reward + gamma * self.V[next_state])
            return A

        while True:
            delta = 0
            for state in range(self.n_states):
                A = compute_expected_reward(state)
                best_action_value = A.max()
                delta = max(delta, np.abs(best_action_value - self.V[state]))
                self.V[state] = best_action_value
            if delta < epslion:
                break

        policy = np.zeros([self.n_states, self.n_actions])
        for state in range(self.n_states):
            A = compute_expected_reward(state)
            best_action = A.argmax()
            policy[state, best_action] = 1.

        return self.V, policy


if __name__ == '__main__':
    env = gym.make('EasyFrozenLakeEnv-v0')
    V, policy = ValueIteration(env.nS, env.nA).run(env, 0.99, 1e-5)

    print('Frozen Lake Env{}'.format(env.render('ansi').getvalue()))

    print('Learned Policy')
    reshaped_policy = policy.argmax(axis=1).reshape(env.desc.shape)
    reshaped_V = V.reshape(env.desc.shape)
    num2arrow = ['←', '↓', '→', '↑']
    for row, v in zip(reshaped_policy, reshaped_V):
        for state, value in zip(row, v):
            if value:
                print(num2arrow[state], end=' ')
            else:
                print('・', end='')
        print()
