import gymnasium as gym
import numpy as np
import time
from gymnasium.wrappers import TimeLimit

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from policy_testing_functions import test_policy, convert_to_arrows, save_table


def show_rewards_per_episodes(rewards, episodes):
    rewards_per_th_ep = np.split(np.array(rewards), episodes/10000)
    count = 10000
    print("=== Rewards per 10000 episodes ===")
    for r in rewards_per_th_ep:
        print(count, ": ", str(sum(r/10000)))
        count += 10000


def q_learning(env, episodes, i_alpha, gamma):
    rewards = []
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    alpha = i_alpha
    epsilon = 1

    for episode in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        rewards_per_episode = 0

        while not terminated and not truncated:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[state])
            else:
                action = env.action_space.sample()

            new_state, reward, terminated, truncated, _ = env.step(action)

            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state, action])

            state = new_state

            if (reward == 1):
                rewards_per_episode += reward

        print(episode, "- Epsilon:", f"{epsilon:.4f}", "- Alpha:", f"{alpha:.4f}") # To visualize if epsilon and alpha are decreasing as intended

        epsilon = max(0.05, 1 - (1 - 0.05) * (episode / episodes))
        alpha = max(0.01, i_alpha * (1 - episode / episodes))

        rewards.append(rewards_per_episode)

    show_rewards_per_episodes(rewards, episodes) # To visualize how the average rewards increase over time

    return q_table


def ql_4x4():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

    env = TimeLimit(env.unwrapped, max_episode_steps=500)

    st = time.time()
    q = q_learning(env, episodes=10000, i_alpha=0.7, gamma=0.99)
    et = time.time()
    
    policy = []
    for i in range(env.observation_space.n):
        best_actions = [a for a, q_val in enumerate(q[i]) if q_val == np.max(q[i])]
        policy.append(best_actions)

    print(test_policy(env, policy))

    policy = []
    for i in range(env.observation_space.n):
        if np.all(q[i] == 0):
            policy.append(None)
        else:
            best_actions = [a for a, q_val in enumerate(q[i]) if q_val == np.max(q[i])]
            policy.append(best_actions)

    arrow_table = convert_to_arrows(policy)
    save_table(arrow_table, 'q_learning/ql_4x4_policy.png', 4)

    print("===============")
    print("Time:", et - st)


def ql_8x8():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

    env = TimeLimit(env.unwrapped, max_episode_steps=500)

    st = time.time()
    q = q_learning(env, episodes=200000, i_alpha=0.7, gamma=0.99)
    et = time.time()
    
    policy = []
    for i in range(env.observation_space.n):
        best_actions = [a for a, q_val in enumerate(q[i]) if q_val == np.max(q[i])]
        policy.append(best_actions)

    print(test_policy(env, policy))

    policy = []
    for i in range(env.observation_space.n):
        if np.all(q[i] == 0):
            policy.append(None)
        else:
            best_actions = [a for a, q_val in enumerate(q[i]) if q_val == np.max(q[i])]
            policy.append(best_actions)

    arrow_table = convert_to_arrows(policy)
    save_table(arrow_table, 'q_learning/ql_8x8_policy.png', 8)

    print("===============")
    print("Time:", et - st)


def main():
    ql_4x4()
    #ql_8x8()
    

if __name__ == "__main__":
    main()