import gymnasium as gym
import numpy as np

import sys
import os
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from policy_testing_functions import convert_to_arrows, save_table, test_policy


def is_action_fully_terminal(env, state, action):
    transitions = env.P[state][action]
    for prob, next_state, reward, terminated in transitions:
        if not terminated:
            return False
    return True


def value_iteration(env):
    theta = 1e-10
    gamma = 1

    value_function = np.zeros(env.observation_space.n)

    while True:
        new_value_function = np.copy(value_function)

        for s in range(env.observation_space.n):
            Q_values = []

            for a in range(env.action_space.n):
                q = 0
                for prob, new_state, reward, _ in env.P[s][a]:
                    q += prob * (reward + gamma * new_value_function[new_state])

                Q_values.append(q)

            if not Q_values:
                continue

            value_function[s] = max(Q_values)

        if (np.sum(np.fabs(new_value_function - value_function)) <= theta):
            break

    return value_function


def extract_policy(env, value_function):
    gamma = 1

    policy = [None] * env.observation_space.n
   
    new_value_function = np.copy(value_function)

    for s in range(env.observation_space.n):
        Q_values = {}

        for a in range(env.action_space.n):
            if is_action_fully_terminal(env, s, a):
                continue

            q = 0
            for prob, new_state, reward, _ in env.P[s][a]:
                q += prob * (reward + gamma * new_value_function[new_state])
                
            Q_values[a] = q

        if not Q_values:
            policy[s] = None
            continue

        max_q = max(Q_values.values())
        best_actions = [a for a, q in Q_values.items() if q == max_q]
        policy[s] = best_actions

    return policy


def vi_4x4():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

    st = time.time()
    value_function = value_iteration(env.unwrapped)
    policy = extract_policy(env.unwrapped, value_function)
    et = time.time()

    print(test_policy(env.unwrapped, policy), "| Time:", et - st)

    arrow_policy = convert_to_arrows(policy)
    save_table(arrow_policy, "value_iteration/vi_4x4_policy.png", 4)


def vi_8x8():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

    st = time.time()
    value_function = value_iteration(env.unwrapped)
    policy = extract_policy(env.unwrapped, value_function)
    et = time.time()

    print(test_policy(env.unwrapped, policy), "| Time:", et - st)

    arrow_policy = convert_to_arrows(policy)
    save_table(arrow_policy, "value_iteration/vi_8x8_policy.png", 8)


def main():
    vi_4x4()
    vi_8x8()


if __name__ == "__main__":
    main()