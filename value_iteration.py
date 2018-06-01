import numpy as np

import gym
from gym.envs.registration import register
register(id='EasyFrozenLakeEnv-v0',  entry_point='gym.envs.toy_text:FrozenLakeEnv',
         kwargs={'is_slippery': False})


class ValueIteration:
    def __init__(self, n_states, n_actions, probs):
        self.n_states = n_states
        self.n_actions = n_actions
        self.probs = probs

    def run(self, gamma, epslion):
        probs = self.probs
        n_states = self.n_states
        n_actions = self.n_actions
        V = np.zeros(n_states)

        def compute_action_value(state):
            A = np.zeros(self.n_actions)
            for action in range(n_actions):
                for prob, next_state, reward, done in probs[state][action]:
                    A[action] += prob * (reward + gamma * V[next_state])
            return A

        while True:
            delta = 0
            for state in range(n_states):
                A = compute_action_value(state)
                best_action_value = A.max()
                delta = max(delta, np.abs(best_action_value - V[state]))
                V[state] = best_action_value
            if delta < epslion:
                break

        policy = np.zeros([n_states, n_actions])
        for state in range(n_states):
            A = compute_action_value(state)
            policy[state] = A
        policy -= policy.max(axis=1, keepdims=True)
        policy = np.exp(policy) / np.exp(policy).sum(axis=1, keepdims=True)
        return V, policy


if __name__ == '__main__':
    env = gym.make('EasyFrozenLakeEnv-v0')
    V, policy = ValueIteration(env.nS, env.nA, env.P).run(0.99, 1e-5)

    print('Frozen Lake Env{}'.format(env.render('ansi').getvalue()))

    print('Learned Policy')
    # reshaped_policy = policy.argmax(axis=1).reshape(env.desc.shape)
    # reshaped_V = V.reshape(env.desc.shape)
    # num2arrow = ['←', '↓', '→', '↑']
    # for row, v in zip(reshaped_policy, reshaped_V):
    #     for state, value in zip(row, v):
    #         if value:
    #             print(num2arrow[state], end=' ')
    #         else:
    #             print('・', end='')
    #     print()

    goal_count = 0
    loop_count = 0
    while goal_count < 10:
        state = env.reset()
        done = False
        while not done:
            action = np.random.multinomial(1, policy[state]).argmax()
            state, reward, done, info = env.step(action)
        if reward == 1:
            goal_count += 1
        loop_count += 1
    print(f'{goal_count/loop_count}')
