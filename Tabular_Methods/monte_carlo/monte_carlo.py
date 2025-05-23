import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import time

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from policy_testing_functions import test_policy, convert_to_arrows, save_table


def monte_carlo(env, num_episodes, gamma):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    n_visits = np.zeros((n_states, n_actions))
    policy = np.zeros(n_states, dtype=int)

    epsilon = 1

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_log = []
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_log.append((state, action, reward))
            state = next_state

        epsilon = max(0.01, 1 - (1 - 0.01) * (episode / num_episodes))
        print(episode, "- Epsilon:", f"{epsilon:.4f}") # To visualize if epsilon is decreasing as intended

        G = 0
        visited = set()
        for t in reversed(range(len(episode_log))):
            s, a, r = episode_log[t]
            G = r + gamma * G
            if (s, a) not in visited:
                visited.add((s, a))
                n_visits[s, a] += 1
                Q[s, a] += (1 / n_visits[s, a]) * (G - Q[s, a])
                policy[s] = np.argmax(Q[s])

    return policy, Q


def mc_4x4():
    env=gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    env=TimeLimit(env.unwrapped, 1000)

    st = time.time()
    policy, q = monte_carlo(env, 350000, 1)
    et = time.time()
    print(test_policy(env, policy), "| Time:", et - st)
    
    zero_row_indices = np.where(np.all(q == 0, axis=1))[0]
    fpolicy = []
    for i, v in enumerate(policy):
        if i in zero_row_indices:
            fpolicy.append(None)
        else:
            fpolicy.append([v])

    a = convert_to_arrows(fpolicy)

    save_table(a, "monte_carlo/mc_4x4_policy", 4)


def mc_8x8():
    env=gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
    env=TimeLimit(env.unwrapped, 1000)

    st = time.time()
    policy, q = monte_carlo(env, 2000000, 1)
    et = time.time()
    print(test_policy(env, policy), "| Time:", et - st)

    zero_row_indices = np.where(np.all(q == 0, axis=1))[0]
    fpolicy = []
    for i, v in enumerate(policy):
        if i in zero_row_indices:
            fpolicy.append(None)
        else:
            fpolicy.append([v])

    a = convert_to_arrows(fpolicy)

    save_table(a, "monte_carlo/mc_8x8_policy", 8)


def main():
    mc_4x4()
    #mc_8x8()


if __name__ == "__main__":
    main()