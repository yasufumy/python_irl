import random

import numpy as np


class ValueIteration:
    def __init__(self, n_states, n_actions, probs):
        self.n_states = n_states
        self.n_actions = n_actions
        self.probs = probs

    def __call__(self, gamma, epslion, reward_function=None):
        probs = self.probs
        n_states = self.n_states
        n_actions = self.n_actions
        V = np.zeros(n_states)

        def compute_action_value(state):
            A = np.zeros(self.n_actions)
            for action in range(n_actions):
                for prob, next_state, reward, done in probs[state][action]:
                    reward = reward if reward_function is None else reward_function(state)
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
        max_values = np.broadcast_to(policy.max(axis=1, keepdims=True), policy.shape)
        policy = np.where(policy == max_values, policy, np.NINF)
        policy = np.exp(policy) / np.exp(policy).sum(axis=1, keepdims=True)
        return V, policy


def sample_trajectories(env, policy, n_steps, n_samples):
    goal_count = 0
    ignore_states = np.bitwise_or.reduce(
        [env.desc.flatten() == s for s in (b'H', b'G')]).nonzero()[0].tolist()
    states = [i for i in range(env.nS) if i not in ignore_states]
    trajectories = []
    while goal_count < n_samples:
        env.reset()
        state = random.choice(states)
        env.s = state
        done = False
        trajectory = []
        for i in range(n_steps):
            action = np.random.multinomial(1, policy[state]).argmax()
            trajectory.append(state)
            state, reward, done, info = env.step(action)
            if done:
                trajectory.extend([state] * (n_steps - len(trajectory)))
                break
        if reward == 1.:
            trajectories.append(trajectory)
            goal_count += 1
    return np.array(trajectories)


if __name__ == '__main__':
    from time import sleep

    import gym
    from gym.envs.registration import register

    register(id='EasyFrozenLakeEnv-v0', entry_point='gym.envs.toy_text:FrozenLakeEnv',
             kwargs={'is_slippery': False})

    env = gym.make('EasyFrozenLakeEnv-v0')
    value_iteration = ValueIteration(env.nS, env.nA, env.P)

    V, policy = value_iteration(0.99, 1e-5)

    state = env.reset()
    done = False
    G = 0
    while not done:
        env.render()
        sleep(1)
        action = np.random.multinomial(1, policy[state]).argmax()
        state, reward, done, _ = env.step(action)
        G += reward
    env.render()
    print(f'Total reward: {G}')
