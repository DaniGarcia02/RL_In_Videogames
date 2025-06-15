import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, AtariPreprocessing, FrameStackObservation
from gymnasium import ActionWrapper
from gymnasium.vector import SyncVectorEnv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import time
import math
import gc

import ale_py

gym.register_envs(ale_py)


def main():

    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
    obs, done = env.reset(), False
    while not done:
        action = env.action_space.sample()   # or a fixed/random policy
        obs, reward, truncated, terminated, info = env.step(action)
    pass


if __name__ == "__main__":
    main()